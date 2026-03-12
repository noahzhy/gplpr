# from memory_profiler import profile
import copy
import time
import yaml
import torch
import torch.nn as nn
import train_funcs
import utils
import random
import argparse
import models
import losses
import smtplib
import datasets
import numpy as np
from tqdm import tqdm

from PIL import Image
from pathlib import Path
from torchvision import transforms
from email.mime.text import MIMEText
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

#Set gpu visibility, for debbug purposes
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Enable anomaly detection only when explicitly requested, because it can significantly slow training.
ENABLE_ANOMALY_DETECTION = os.environ.get('GPLPR_DETECT_ANOMALY', '0') == '1'
torch.autograd.set_detect_anomaly(ENABLE_ANOMALY_DETECTION)

# Set the per-process GPU memory fraction to 90%. This means that the GPU will allocate a maximum
# of 90% of its available memory for this process. This can be useful to limit the GPU memory usage
# when running multiple processes or to prevent running out of GPU memory.
torch.cuda.set_per_process_memory_fraction(1.0, 0)

# Clear the GPU memory cache. This releases GPU memory that is no longer in use and can help free up
# memory for other operations. It's particularly useful when working with limited GPU memory.
torch.cuda.empty_cache()

DEFAULT_ALPHABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def _build_loader_kwargs(dataset, spec, tag, num_workers, pin_memory, persistent_workers, prefetch_factor):
    loader_kwargs = {
        'dataset': dataset,
        'batch_size': spec['batch'],
        'shuffle': (tag == 'train'),
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'collate_fn': dataset.collate_fn,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = persistent_workers
        loader_kwargs['prefetch_factor'] = prefetch_factor
    return loader_kwargs


def _auto_tune_loader(dataset, spec, tag, pin_memory):
    # Keep startup overhead bounded while still profiling realistic loading throughput.
    warmup_batches = int(spec.get('tune_batches', 8))
    cpu_count = os.cpu_count() or 1
    default_candidates = [0, 2, 4, 8, min(12, cpu_count)]
    worker_candidates = spec.get('tune_num_workers_candidates', default_candidates)
    worker_candidates = sorted(set(int(x) for x in worker_candidates if 0 <= int(x) <= cpu_count))
    if not worker_candidates:
        worker_candidates = [0]

    prefetch_candidates = spec.get('tune_prefetch_candidates', [2, 4])
    prefetch_candidates = sorted(set(max(1, int(x)) for x in prefetch_candidates))

    best = None
    for nw in worker_candidates:
        pf_values = prefetch_candidates if nw > 0 else [None]
        for pf in pf_values:
            kwargs = _build_loader_kwargs(
                dataset=dataset,
                spec=spec,
                tag=tag,
                num_workers=nw,
                pin_memory=pin_memory,
                persistent_workers=(nw > 0),
                prefetch_factor=2 if pf is None else pf,
            )
            try:
                loader = DataLoader(**kwargs)
                t0 = time.perf_counter()
                seen = 0
                for seen, _ in enumerate(loader, start=1):
                    if seen >= warmup_batches:
                        break
                elapsed = max(1e-6, time.perf_counter() - t0)
                throughput = seen / elapsed
                del loader
                if best is None or throughput > best['throughput']:
                    best = {'num_workers': nw, 'prefetch_factor': pf, 'throughput': throughput}
            except Exception:
                continue

    if best is None:
        return None

    if 'log' in globals() and callable(log):
        log(
            'Auto-tuned dataloader ({}): num_workers={}, prefetch_factor={}, throughput={:.2f} batch/s'.format(
                tag,
                best['num_workers'],
                best['prefetch_factor'] if best['prefetch_factor'] is not None else '-',
                best['throughput'],
            )
        )
    return best


def _get_alphabet(config):
    alphabet = config.get('alphabet')
    if alphabet is not None:
        return alphabet

    model_args = config.get('model', {}).get('args', {})
    alphabet = model_args.get('alphabet')
    if alphabet is not None:
        return alphabet

    return DEFAULT_ALPHABET


def _build_stage_spec(stage_config, alphabet, default_k, batch_size, with_lr=False):
    wrapper_args = {
        'alphabet': alphabet,
        'maxT': stage_config.get('maxT', default_k),
        'img_size': stage_config.get('img_size', [32, 96]),
        'data_aug': stage_config.get('data_aug', False),
        'image_dir': stage_config.get('image_dir'),
        'language': stage_config.get('language'),
        'with_lr': with_lr,
    }

    stage_spec = {
        'wrapper': {
            'name': 'Ocr_images_lp',
            'args': wrapper_args,
        },
        'batch': batch_size,
    }

    if stage_config.get('path_split') is not None:
        stage_spec['dataset'] = {
            'name': 'ocr_img',
            'args': {
                'path_split': stage_config['path_split'],
                'phase': stage_config.get('phase', 'training'),
            },
        }

    return stage_spec


def normalize_config(config):
    if 'train_dataset' in config and 'val_dataset' in config:
        return config

    if 'train' not in config or 'val' not in config:
        return config

    normalized = copy.deepcopy(config)
    alphabet = _get_alphabet(normalized)
    time_steps = normalized.get('time_steps', normalized.get('train', {}).get('maxT', 9))
    batch_size = normalized.get('batch_size', 128)
    model_input_shape = normalized.get('input_shape', [32, 96, 1])
    model_channels = 3 if len(model_input_shape) < 3 else max(3, int(model_input_shape[2]))

    normalized['alphabet'] = alphabet
    normalized.setdefault('func_train', 'GP_LPR_TRAIN')
    normalized.setdefault('func_val', 'GP_LPR_VAL')
    normalized.setdefault('optimizer', {
        'name': 'adam',
        'args': {
            'lr': normalized.get('lr', 1.e-3),
            'betas': [0.5, 0.555],
        },
    })
    normalized.setdefault('loss', {
        'name': 'CrossEntropyLoss',
        'args': {
            'size_average': None,
            'reduce': None,
            'reduction': 'mean',
        },
    })
    normalized.setdefault('early_stopper', {
        'patience': 400,
        'min_delta': 0,
        'counter': 0,
    })
    normalized.setdefault('epoch_max', normalized.get('epochs', 3000))
    normalized.setdefault('epoch_save', normalized.get('eval_freq', 100))
    normalized.setdefault('resume', None)
    normalized.setdefault('use_amp', torch.cuda.is_available())
    normalized.setdefault('amp_dtype', 'float16')  # float16 or bfloat16
    normalized.setdefault('use_channels_last', torch.cuda.is_available())
    normalized.setdefault('use_torch_compile', False)
    normalized.setdefault('torch_compile_mode', 'reduce-overhead')
    normalized.setdefault('torch_compile_backend', None)
    normalized.setdefault('auto_tune_loader', True)
    normalized.setdefault('loader_tune_batches', 8)
    normalized.setdefault('model', {
        'name': 'GPLPR',
        'OCR_TRAIN': True,
        'args': {
            'nc': model_channels,
            'alphabet': alphabet,
            'k': time_steps,
            'isSeqModel': True,
            'head': 2,
            'inner': 256,
            'isl2Norm': True,
        },
    })

    normalized['train_dataset'] = _build_stage_spec(
        normalized['train'], alphabet, time_steps, batch_size, with_lr=normalized.get('with_lr', False)
    )
    normalized['train_dataset'].setdefault('auto_tune_loader', normalized.get('auto_tune_loader', True))
    normalized['train_dataset'].setdefault('tune_batches', normalized.get('loader_tune_batches', 8))
    normalized['val_dataset'] = _build_stage_spec(
        normalized['val'], alphabet, time_steps, batch_size, with_lr=False
    )
    normalized['val_dataset'].setdefault('auto_tune_loader', False)
    normalized['val_dataset'].setdefault('tune_batches', max(2, normalized.get('loader_tune_batches', 8) // 2))
    if 'test' in normalized and 'test_dataset' not in normalized:
        normalized['test_dataset'] = _build_stage_spec(
            normalized['test'], alphabet, time_steps, batch_size, with_lr=False
        )

    return normalized

def make_dataloader(spec, tag=''):
    dataset = None
    if spec.get('dataset') is not None:
        dataset = datasets.make(spec['dataset'])

    wrapper_args = {'dataset': dataset} if dataset is not None else None
    dataset = datasets.make(spec['wrapper'], args=wrapper_args)
    
    num_workers = int(spec.get('num_workers', min(8, os.cpu_count() or 1)))
    pin_memory = bool(spec.get('pin_memory', torch.cuda.is_available()))
    persistent_workers = bool(spec.get('persistent_workers', num_workers > 0))
    prefetch_factor = int(spec.get('prefetch_factor', 2))

    if spec.get('auto_tune_loader', False):
        tuned = _auto_tune_loader(dataset, spec, tag, pin_memory)
        if tuned is not None:
            num_workers = tuned['num_workers']
            if tuned['prefetch_factor'] is not None:
                prefetch_factor = tuned['prefetch_factor']
            persistent_workers = num_workers > 0

    loader_kwargs = _build_loader_kwargs(
        dataset=dataset,
        spec=spec,
        tag=tag,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    loader = DataLoader(**loader_kwargs)

    # """Next is only for debugging purpose"""
    # for batch in loader:
    #     a=0
    # Return the DataLoader        
    return loader 

def make_dataloaders():
    # Create data loaders for the training and validation datasets
    # These data loaders are typically created using custom functions (e.g., make_dataloader)
    train_loader = make_dataloader(config['train_dataset'], tag='train')
    val_loader = make_dataloader(config['val_dataset'], tag='val')

    # Return the created data loaders
    return train_loader, val_loader
    
def prepare_training():
    # Check if a training checkpoint is specified in the configuration (resuming training)
    if config.get('resume') is not None:
        # Load the saved checkpoint file
        sv_file = torch.load(config['resume'])
        
        # Create the model using the configuration from the checkpoint and move it to the GPU
        model = models.make(sv_file['model'], load_model=True).cuda()
        
        # Create an optimizer with parameters from the checkpoint and load its state
        optimizer = utils.make_optimizer(model.parameters(), sv_file['optimizer'], load_optimizer=True)
        
        # Create an EarlyStopping object using settings from the checkpoint
        early_stopper = utils.Early_stopping(**sv_file['early_stopping'])
        
        # Get the starting epoch from the checkpoint and set the random number generator state
        epoch_start = sv_file['epoch'] + 1     
        state = sv_file['state']                
        torch.set_rng_state(state)
        
        # Print a message indicating that training is resuming
        print(f'Resuming from epoch {epoch_start}...')
        log(f'Resuming from epoch {epoch_start}...')
        
        # Check if a learning rate scheduler (ReduceLROnPlateau) is specified in the configuration
        if config.get('reduce_on_plateau') is None:
            lr_scheduler = None
        else:
            lr_scheduler = ReduceLROnPlateau(optimizer, **config['reduce_on_plateau'])
        
        # Set the learning rate scheduler's last_epoch to the resumed epoch
        lr_scheduler.last_epoch = epoch_start - 1
       
    # If no checkpoint is specified, start training from scratch
    else:
        print('Training from start...')
        
        # Create the model using the configuration and move it to the GPU
        model = models.make(config['model']).cuda()
        
        # Create an optimizer using the configuration
        optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
        # Create an EarlyStopping object using settings from the configuration
        early_stopper = utils.Early_stopping(**config['early_stopper'])

        # Set the starting epoch to 1
        epoch_start = 1
        
        # Check if a learning rate scheduler (ReduceLROnPlateau) is specified in the configuration
        if config.get('reduce_on_plateau') is None:
            lr_scheduler = None
        else:
            # Create a learning rate scheduler using settings from the configuration
            lr_scheduler = ReduceLROnPlateau(optimizer, **config['reduce_on_plateau'])
            #StepLR(optimizer, step_size=50, gamma=0.8, verbose=True)
            # lr_scheduler = ReduceLROnPlateau(optimizer, **config['reduce_on_plateau'])
            
        # For epochs prior to the starting epoch, step the learning rate scheduler
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
            
    if config.get('use_channels_last', torch.cuda.is_available()):
        model = model.to(memory_format=torch.channels_last)

    if config.get('use_torch_compile', False) and hasattr(torch, 'compile'):
        compile_mode = config.get('torch_compile_mode', 'reduce-overhead')
        compile_backend = config.get('torch_compile_backend')
        try:
            if compile_backend:
                model = torch.compile(model, mode=compile_mode, backend=compile_backend)
            else:
                model = torch.compile(model, mode=compile_mode)
            log(f'torch.compile enabled (mode={compile_mode}, backend={compile_backend})')
        except Exception as e:
            log(f'torch.compile disabled due to error: {e}')

    # Log the number of model parameters and model structure
    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    log('model: #struct={}'.format(model))
    
    # Return the model, optimizer, starting epoch, learning rate scheduler, and EarlyStopping object
    return model, optimizer, epoch_start, lr_scheduler, early_stopper

def main(config_, save_path):
    # Declare global variables
    global config, log, writer
    config = config_
    
    # Create log and writer for logging training progress
    log, writer = utils.make_log_writer(save_path)

    # Create data loaders for training and validation datasets
    train_loader, val_loader = make_dataloaders()

    # Initialize the model, optimizer, learning rate scheduler, and early stopper
    model, optimizer, epoch_start, lr_scheduler, early_stopper = prepare_training()
    train = train_funcs.make(config['func_train'])
    validation = train_funcs.make(config['func_val'])

    use_amp = bool(config.get('use_amp', torch.cuda.is_available())) and torch.cuda.is_available()
    amp_dtype = str(config.get('amp_dtype', 'float16')).lower()
    scaler_enabled = use_amp and amp_dtype in ('float16', 'fp16')
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    # Create the loss function for training    
    loss_fn = losses.make(config['loss'])

    # Get the number of available GPUs
    n_gpus = torch.cuda.device_count()
    print(n_gpus)

    # If multiple GPUs are available, use DataParallel to parallelize model training
    # if n_gpus > 1:
    #     model = nn.parallel.DataParallel(model)

    # Get maximum number of epochs and epoch save interval from configuration
    epoch_max = config['epoch_max']
    epoch_save = config.get('epoch_save', 1)  # 默认每1个epoch保存一次

    # 中间保存目录，默认当前路径下 save 文件夹
    intermediate_save_path = Path('./save')
    intermediate_save_path.mkdir(parents=True, exist_ok=True)

    # Create a timer to measure training time
    timer = utils.Timer()  
    confusing_pair = []
    # Loop over epochs for training
    for epoch in range(epoch_start, epoch_max+1):
        # Initialize timer for the current epoch
        print(f"epoch {epoch}/{epoch_max}")
        t_epoch_init = timer._get()

        # Prepare logging information for the current epoch
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        # Log the learning rate and add it to the writer
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        log_info.append('lr:{}'.format(optimizer.param_groups[0]['lr']))

        # Perform training for the current epoch and get the training loss
        train_loss = train(train_loader, model, optimizer, loss_fn, confusing_pair, config, scaler) 
        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalar('train_loss', train_loss, epoch)

        # Perform validation for the current epoch and get the validation loss
        val_loss, confusing_pair = validation(val_loader, model, loss_fn, confusing_pair, config)             
        log_info.append('val: loss={:.4f}'.format(val_loss))
        writer.add_scalar('val_train_loss', val_loss, epoch)

        # Adjust the learning rate using the learning rate scheduler if it's defined
        if lr_scheduler is not None:
            lr_scheduler.step(val_loss)

        # Calculate and log elapsed times for the current epoch
        t = timer._get()        
        t_epoch = timer.time_text(t - t_epoch_init )
        t_elapsed = timer.time_text(t)
        log_info.append('{} / {}'.format(t_epoch, t_elapsed))

        # Check for early stopping and log the status
        stop, bm = early_stopper.early_stop(val_loss)
        log_info.append('Early stop {} / Best model {}'.format(stop, bm))

        # Get the underlying model (without DataParallel) if multiple GPUs are used
        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model

        # Prepare model and optimizer specifications for saving
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        early_stopper_ = vars(early_stopper)

        # Get the current random number generator state
        state = torch.get_rng_state()

        # Create a dictionary to save the model checkpoint
        sv_file = {
            'model': model_spec, 
            'optimizer': optimizer_spec, 
            'epoch': epoch, 
            'state': state, 
            'early_stopping': early_stopper_
            }
        if scaler.is_enabled():
            sv_file['scaler'] = scaler.state_dict()

        # Save the model checkpoint if it's the best model so far
        if bm:
            torch.save(sv_file, save_path / Path('best_model_'+'Epoch_{}'.format(epoch)+'.pth'))
        else:
            torch.save(sv_file, save_path / Path('epoch-last.pth'))

        # Save the model checkpoint if it's an epoch save interval (主保存)
        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file, save_path / Path('epoch-{}.pth'.format(epoch)))

        # 中间保存：每个epoch都保存到 ./save 目录下
        torch.save(sv_file, intermediate_save_path / Path(f'intermediate_epoch_{epoch}.pth'))

        # Log the training progress for the current epoch
        log(', '.join(log_info))
        writer.flush()

        # Check for early stopping and break the loop if early stopping criteria are met
        if stop:
            print('Early stop: {}'.format(stop))
            break


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--save', default=None)    
    parser.add_argument('--tag', default=None)
    args = parser.parse_args()
    
    
    # Define a function to set random seeds for reproducibility
    def setup_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # sets the seed for cpu
        torch.cuda.manual_seed(seed)  # Sets the seed for the current GPU.
        torch.cuda.manual_seed_all(seed)  #  Sets the seed for the all GPU.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    
    # Set a fixed random seed (for reproducibility)
    setup_seed(1996)
    
    # Read the configuration file (usually in YAML format)
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = normalize_config(config)
    
    # Determine the save name for checkpoints
    save_name = args.save
    if save_name is not None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
        
    # Create a save path for model checkpoints and ensure the directory exists
    save_path = Path("/home/haoyu/projects/gplpr/save")
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Call the main training function with the configuration and save path
    main(config, save_path)
    
    # Send an email notification about training completion (optional)
    if config.get('email') is not None:
        msg = MIMEText("Your training process has completed successfully.")
        msg['Subject'] = config['email']['subject'] + '_' + config['model']['name']
        msg['From'] = config['email']['sender']
        msg['To'] = config['email']['recipient']
        
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(config['email']['sender'], config['email']['passwd'])
        server.sendmail(config['email']['sender'], config['email']['recipient'], msg.as_string())
        server.quit()
