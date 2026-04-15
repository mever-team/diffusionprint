import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import json
import csv
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from builders import DiffusionPrint
from dataset import DiffusionPrintDataset


def parse_args():
    parser = argparse.ArgumentParser(description='DiffusionPrint Training')

    # Data
    parser.add_argument('--train_csv', type=str)
    parser.add_argument('--patches_dir', type=str)
    parser.add_argument('--exclude_generators', type=str, nargs='+', default=None,
                        metavar='GEN',
                        help='Generator models to exclude from training. '
                             'Valid values: none, sd2, sdxl, flux')

    # Mode
    parser.add_argument('--mode', type=str, default='projector',
                        choices=['projector', 'flatten_projector'],
                        help='projector: GAP+MLP+cosine | '
                             'flatten_projector: flatten+MLP+cosine')

    # DnCNN
    parser.add_argument('--num_levels', type=int, default=17)
    parser.add_argument('--hidden_features', type=int, default=64)
    parser.add_argument('--out_channels', type=int, default=256,
                        help='DnCNN output channels (forced to 1 in flatten_projector)')

    # Contrastive
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--projector_hidden_dim', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--augmentations', type=str, default='none',
                        choices=['none', 'geometric', 'mild', 'strong'])
    parser.add_argument('--queue_size', type=int, default=65536)
    parser.add_argument('--moco_momentum', type=float, default=0.999)
    parser.add_argument('--top_k', type=int, default=64)
    parser.add_argument('--mining_mode', type=str, default='binary',
                        choices=['binary', 'multiclass'])
    parser.add_argument('--neg_batch_size', type=int, default=128)

    # Classification head
    parser.add_argument('--cls_lambda', type=float, default=0.0)
    parser.add_argument('--num_classes', type=int, default=2)

    # Frequency filtering
    parser.add_argument('--input_highpass', action='store_true', default=False,
                        help='Apply fixed Gaussian high-pass to ALL inputs before backbone.')
    parser.add_argument('--input_highpass_sigma', type=float, default=1.0,
                        help='Gaussian sigma for input high-pass filter (default: 1.0)')
    parser.add_argument('--anchor_bandpass', action='store_true', default=False,
                        help='Apply asymmetric bandpass to anchor only.')
    parser.add_argument('--bandpass_low_sigma', type=float, default=3.0,
                        help='Low-frequency Gaussian sigma for bandpass (default: 3.0)')
    parser.add_argument('--bandpass_high_sigma', type=float, default=0.5,
                        help='High-frequency Gaussian sigma for bandpass (default: 0.5)')

    # Legacy high-pass (deprecated, kept for backwards compat)
    parser.add_argument('--highpass_prob', type=float, default=0.0,
                        help='[Deprecated] Probabilistic Laplacian high-pass. Use --input_highpass instead.')

    # Pretrained weights
    parser.add_argument('--noiseprint_weights', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None)

    # Training
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_step', type=int, default=10)
    parser.add_argument('--grad_clip', type=float, default=10.0)
    parser.add_argument('--num_workers', type=int, default=8)

    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='./ckpt')
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()
    return args


def save_checkpoint(model, optimizer, scheduler, epoch, args, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'args': vars(args),
    }
    save_path = Path(args.save_dir) / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Resumed from epoch {epoch}")
    return epoch


def save_params(args, save_dir, total_params, random_baseline, dataset_size):
    params = vars(args).copy()
    params['trainable_parameters'] = total_params
    params['random_baseline'] = random_baseline
    params['dataset_size'] = dataset_size
    params['timestamp'] = datetime.now().isoformat()
    params['cuda_device'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    params['pytorch_version'] = torch.__version__

    json_path = Path(save_dir) / 'params.json'
    with open(json_path, 'w') as f:
        json.dump(params, f, indent=2, default=str)

    txt_path = Path(save_dir) / 'params.txt'
    with open(txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DiffusionPrint Training Configuration\n")
        f.write("=" * 60 + "\n\n")

        sections = {
            'Mode': ['mode', 'augmentations'],
            'Architecture': ['num_levels', 'hidden_features', 'out_channels',
                             'projection_dim', 'projector_hidden_dim', 'trainable_parameters'],
            'Contrastive': ['temperature', 'queue_size', 'moco_momentum', 'top_k',
                            'mining_mode', 'neg_batch_size', 'random_baseline'],
            'Classification': ['cls_lambda', 'num_classes'],
            'Frequency Filtering': ['input_highpass', 'input_highpass_sigma',
                                    'anchor_bandpass', 'bandpass_low_sigma',
                                    'bandpass_high_sigma', 'highpass_prob'],
            'Training': ['batch_size', 'epochs', 'lr', 'weight_decay', 'lr_step',
                         'grad_clip', 'num_workers'],
            'Data': ['train_csv', 'patches_dir', 'exclude_generators', 'dataset_size'],
            'Weights': ['noiseprint_weights', 'pretrained', 'resume'],
            'Checkpointing': ['save_dir', 'save_freq'],
            'System': ['cuda_device', 'pytorch_version', 'timestamp'],
        }

        for section, keys in sections.items():
            f.write(f"[{section}]\n")
            for k in keys:
                if k in params:
                    f.write(f"  {k}: {params[k]}\n")
            f.write("\n")

    print(f"Saved params to {txt_path}")


def init_loss_csv(save_dir, has_cls):
    csv_path = Path(save_dir) / 'losses.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['epoch', 'avg_loss', 'min_loss', 'max_loss',
                  'nan_batches', 'valid_batches', 'lr', 'epoch_time_sec']
        if has_cls:
            header.extend(['avg_instance_loss', 'avg_cls_loss', 'cls_acc'])
        writer.writerow(header)
    return csv_path


def log_epoch(csv_path, epoch, avg_loss, min_loss, max_loss,
              nan_count, valid_batches, lr, epoch_time,
              avg_instance_loss=None, avg_cls_loss=None, cls_acc=None,
              has_cls=False):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        row = [epoch, f'{avg_loss:.6f}', f'{min_loss:.6f}', f'{max_loss:.6f}',
               nan_count, valid_batches, f'{lr:.8f}', f'{epoch_time:.1f}']
        if has_cls and avg_instance_loss is not None:
            row.extend([f'{avg_instance_loss:.6f}', f'{avg_cls_loss:.6f}', f'{cls_acc:.4f}'])
        writer.writerow(row)

if __name__ == '__main__'
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    has_cls = args.cls_lambda > 0

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cls_tag = f"_cls{args.cls_lambda}" if has_cls else ""
    nc_tag = f"_nc{args.num_classes}" if args.num_classes > 2 else ""
    mining_tag = f"_mine{args.mining_mode}" if args.mining_mode != 'binary' else ""
    np_tag = "_npinit" if args.noiseprint_weights else ""
    excl_tag = "_excl{}".format("-".join(sorted(args.exclude_generators))) if args.exclude_generators else ""
    hp_tag = "_ihp" if args.input_highpass else ""
    if args.input_highpass:
        hp_tag += f"s{args.input_highpass_sigma}"
    bp_tag = "_abp" if args.anchor_bandpass else ""
    if args.anchor_bandpass:
        bp_tag += f"l{args.bandpass_low_sigma}h{args.bandpass_high_sigma}"
    legacy_hp_tag = f"_hp{args.highpass_prob}" if args.highpass_prob > 0 else ""

    if args.mode == 'projector':
        run_name = (f"{args.mode}_oc{args.out_channels}_proj{args.projector_hidden_dim}-{args.projection_dim}"
                    f"_t{args.temperature}_bs{args.batch_size}_lr{args.lr}"
                    f"_q{args.queue_size}_topk{args.top_k}_m{args.moco_momentum}"
                    f"_gc{args.grad_clip}_wd{args.weight_decay}_aug{args.augmentations}"
                    f"{hp_tag}{bp_tag}{legacy_hp_tag}{cls_tag}{nc_tag}{mining_tag}"
                    f"{np_tag}{excl_tag}_{timestamp}")
    else:
        run_name = (f"{args.mode}_t{args.temperature}_bs{args.batch_size}_lr{args.lr}"
                    f"_q{args.queue_size}_topk{args.top_k}_m{args.moco_momentum}"
                    f"_gc{args.grad_clip}_wd{args.weight_decay}_aug{args.augmentations}"
                    f"{hp_tag}{bp_tag}{legacy_hp_tag}{cls_tag}{nc_tag}{mining_tag}"
                    f"{np_tag}{excl_tag}_{timestamp}")

    args.save_dir = str(Path(args.save_dir) / run_name)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints: {args.save_dir}")

    model = DiffusionPrint(
        mode=args.mode,
        num_levels=args.num_levels,
        hidden_features=args.hidden_features,
        out_channels=args.out_channels,
        projection_dim=args.projection_dim,
        projector_hidden_dim=args.projector_hidden_dim,
        temperature=args.temperature,
        queue_size=args.queue_size,
        momentum=args.moco_momentum,
        top_k=args.top_k,
        image_size=64,
        augmentations=args.augmentations,
        cls_lambda=args.cls_lambda,
        num_classes=args.num_classes,
        highpass_prob=args.highpass_prob,
        mining_mode=args.mining_mode,
        input_highpass=args.input_highpass,
        input_highpass_sigma=args.input_highpass_sigma,
        anchor_bandpass=args.anchor_bandpass,
        bandpass_low_sigma=args.bandpass_low_sigma,
        bandpass_high_sigma=args.bandpass_high_sigma,
    )

    if args.noiseprint_weights:
        model.load_noiseprint_weights(args.noiseprint_weights)

    if args.pretrained:
        ckpt = torch.load(args.pretrained, map_location='cpu')
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {args.pretrained}")
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Mode: {args.mode}")
    print(f"  DnCNN: {args.num_levels} layers, {args.hidden_features} features, "
          f"out_channels={model.out_channels}")
    if args.mode == 'projector':
        print(f"Projector: GAP({model.out_channels}) -> {args.projector_hidden_dim} -> {args.projection_dim}")
    print(f"Temperature: {args.temperature}")
    print(f"Augmentations: {args.augmentations}")
    if args.input_highpass:
        print(f"Input high-pass: Gaussian sigma={args.input_highpass_sigma} (all inputs, fixed)")
    if args.anchor_bandpass:
        print(f"Anchor bandpass: low_sigma={args.bandpass_low_sigma}, "
              f"high_sigma={args.bandpass_high_sigma} (anchor only)")
    if args.highpass_prob > 0:
        print(f"[Deprecated] Legacy high-pass prob={args.highpass_prob}")
    if has_cls:
        print(f"Classification head: Linear({args.projection_dim}, {args.num_classes}), lambda={args.cls_lambda}")
    print(f"Queue: size={args.queue_size}, momentum={args.moco_momentum}, top_k={args.top_k}")
    print(f"Mining mode: {args.mining_mode}")
    print(f"Trainable params: {total_params:,}")

    random_baseline = torch.log(torch.tensor(float(1 + args.batch_size + args.top_k - 1))).item()
    print(f"Random baseline: log(1+B+top_k-1) = {random_baseline:.4f}")

    transform = transforms.ToTensor()

    train_dataset = DiffusionPrintDataset(
        csv_path=args.train_csv,
        dat_path=args.patches_dir,
        transform=transform,
        exclude_generators=args.exclude_generators,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume)

    save_params(args, args.save_dir, total_params, random_baseline, len(train_dataset))
    loss_csv_path = init_loss_csv(args.save_dir, has_cls)

    print("Pre-filling queue...")
    model.eval()
    filled = 0
    pbar_q = tqdm(total=args.queue_size, desc="Pre-filling queue", unit="emb")
    while filled < args.queue_size:
        n = min(args.neg_batch_size, args.queue_size - filled)
        neg_imgs, neg_cats, neg_gens = train_dataset.sample_neg_batch(n)
        neg_imgs = neg_imgs.to(device)
        neg_cats = neg_cats.to(device)
        neg_gens = neg_gens.to(device)
        with torch.no_grad():
            neg_z = model._encode_k(neg_imgs)
        model._enqueue(neg_z, neg_cats, neg_gens)
        filled += n
        pbar_q.update(n)
    pbar_q.close()
    print(f"Queue pre-filled with {filled} embeddings")

    model.train()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        total_loss = 0.0
        total_instance_loss = 0.0
        total_cls_loss = 0.0
        total_cls_correct = 0
        total_cls_samples = 0
        min_loss = float('inf')
        max_loss = float('-inf')
        nan_count = 0
        valid_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}', leave=True)
        for i, (anchors, positives, cats, gen_labels) in enumerate(pbar):
            anchors = anchors.to(device)
            positives = positives.to(device)
            cats = cats.to(device)
            gen_labels = gen_labels.to(device)

            neg_imgs, neg_cats, neg_gens = train_dataset.sample_neg_batch(args.neg_batch_size)
            neg_imgs = neg_imgs.to(device)
            neg_cats = neg_cats.to(device)
            neg_gens = neg_gens.to(device)

            optimizer.zero_grad()

            output = model(anchors, positives, cats, neg_imgs, neg_cats,
                           anchor_gen_labels=gen_labels, neg_gen_labels=neg_gens)

            # Unpack: scalar (instance only) or (total, instance, cls, cls_acc)
            if isinstance(output, tuple):
                loss, instance_loss, cls_loss, cls_acc_batch = output
            else:
                loss = output
                instance_loss = None
                cls_loss = None
                cls_acc_batch = None

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                pbar.set_postfix(loss='NaN', nan=nan_count)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            loss_val = loss.item()
            total_loss += loss_val
            valid_batches += 1
            min_loss = min(min_loss, loss_val)
            max_loss = max(max_loss, loss_val)
            avg_loss = total_loss / valid_batches

            postfix = {'loss': f'{loss_val:.4f}', 'avg': f'{avg_loss:.4f}'}
            if instance_loss is not None:
                total_instance_loss += instance_loss.item()
                postfix['inst'] = f'{instance_loss.item():.4f}'
            if cls_loss is not None:
                total_cls_loss += cls_loss.item()
                total_cls_correct += cls_acc_batch.item() * cats.shape[0]
                total_cls_samples += cats.shape[0]
                postfix['cls'] = f'{cls_loss.item():.4f}'
                postfix['acc'] = f'{total_cls_correct / total_cls_samples:.3f}'
            pbar.set_postfix(**postfix)

        epoch_time = time.time() - epoch_start
        avg_epoch_loss = total_loss / max(valid_batches, 1)
        current_lr = scheduler.get_last_lr()[0]

        parts = [f'Epoch {epoch+1} | Loss: {avg_epoch_loss:.4f}']
        loss_details = []
        if total_instance_loss > 0:
            avg_inst = total_instance_loss / max(valid_batches, 1)
            loss_details.append(f'inst={avg_inst:.4f}')
        if has_cls and total_cls_loss > 0:
            avg_cls = total_cls_loss / max(valid_batches, 1)
            cls_acc = total_cls_correct / max(total_cls_samples, 1)
            loss_details.append(f'cls={avg_cls:.4f}')
            loss_details.append(f'cls_acc={cls_acc:.3f}')
        if loss_details:
            parts.append(f' ({", ".join(loss_details)})')
        parts.append(f' | Baseline: {random_baseline:.4f} | NaN: {nan_count}')
        parts.append(f' | LR: {current_lr:.6f} | Time: {epoch_time:.0f}s')
        print(''.join(parts))

        log_epoch(loss_csv_path, epoch + 1, avg_epoch_loss,
                  min_loss if min_loss != float('inf') else 0.0,
                  max_loss if max_loss != float('-inf') else 0.0,
                  nan_count, valid_batches, current_lr, epoch_time,
                  avg_instance_loss=total_instance_loss / max(valid_batches, 1) if has_cls else None,
                  avg_cls_loss=total_cls_loss / max(valid_batches, 1) if has_cls else None,
                  cls_acc=total_cls_correct / max(total_cls_samples, 1) if has_cls else None,
                  has_cls=has_cls)

        scheduler.step()

        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(model, optimizer, scheduler, epoch + 1, args,
                            f'checkpoint_epoch_{epoch+1}.pth')

        save_checkpoint(model, optimizer, scheduler, epoch + 1, args,
                        'checkpoint_latest.pth')

    print("Training completed!")
    save_checkpoint(model, optimizer, scheduler, args.epochs, args, 'checkpoint_final.pth')
