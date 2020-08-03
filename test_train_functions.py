import argparse
from timeit import default_timer as timer
import os
import time

import torch
import torch.optim as optim
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision.utils as vutils
from tqdm import tqdm

from utils import one_hot_encode, capsnet_training_loss, capsnet_testing_loss, accuracy


def full_precision_training(model, decoder, num_classes, data_loader, optimizer, scheduler, curr_epoch, hard, args):
    """ Full precision training of the model

    Args:
        model  :  model to be trained  (nn.Module)
        decoder : decoder used for training (nn.Module)
        num_classes  :  number of classes of the dataset  (int)
        data_loader  :  training DataLoader (see data_loaders.py)
        optimizer  :  optimizer (e.g. torch.optim.Adam())
        scheduler  :  learning rate scheduler (e.g. torch.optim.lr_scheduler.LambdaLR())
        curr_epoch  :  current training epoch (int)
        hard: bool to toggle hard training
        args  :  arguments set by user in main """

    print('===> Training mode')

    num_batches = len(data_loader)  # number of batches to be processed
    total_step = args.epochs * num_batches  # total number of training steps
    epoch_tot_acc = 0

    # Switch to train mode
    model.train()
    decoder.train()

    # Start timer
    start_time = timer()

    # Iterate over all the batches
    for batch_idx, (data, target) in enumerate(tqdm(data_loader, unit='batch')):
        batch_size = data.size(0)
        global_step = batch_idx + (curr_epoch * num_batches) - num_batches  # current (total) training step

        target_one_hot = one_hot_encode(target, length=num_classes)  # Dataset labels in one-hot encoding

        data, target_one_hot = Variable(data), Variable(target_one_hot)

        if torch.cuda.device_count() > 0:  # if there are available GPUs, move data to the first visible
            device = torch.device("cuda:0")
            data = data.to(device)
            target = target.to(device)
            target_one_hot = target_one_hot.to(device)

        # TRAINING STEP
        optimizer.zero_grad()

        # FORWARD
        output = model(data)  # output from DigitCaps (out_digit_caps)
        reconstruction = decoder(output, target)
        loss, margin_loss, recon_loss = \
            capsnet_training_loss(output, target_one_hot, args.regularization_scale, reconstruction,
                                  data.view(batch_size, -1), hard)

        # BACKWARD
        loss.backward()

        # UPDATE PARAMETERS AND LEARNING RATE
        optimizer.step()
        scheduler.step()

        # Calculate accuracy for each step and average accuracy for each epoch
        acc = accuracy(output, target)
        epoch_tot_acc += acc
        epoch_avg_acc = epoch_tot_acc / (batch_idx + 1)

        # Print losses
        if batch_idx % args.log_interval == 0:
            template = 'Epoch {}/{}, ' \
                       'Step {}/{}: ' \
                       '[Total loss: {:.6f},' \
                       '\tMargin loss: {:.6f},' \
                       '\tReconstruction loss: {:.6f},' \
                       '\tBatch accuracy: {:.6f},' \
                       '\tAccuracy: {:.6f}]'
            tqdm.write(template.format(
                curr_epoch,
                args.epochs,
                global_step,
                total_step,
                loss.data.item(),
                margin_loss.data.item(),
                recon_loss.data.item(),
                acc,
                epoch_avg_acc))

    # Print time elapsed for an epoch
    end_time = timer()
    print('Time elapsed for epoch {}: {:.0f}s.'.format(curr_epoch, end_time - start_time))


def full_precision_test(model, num_classes, data_loader, model_filename, best_accuracy, save_model):
    """ Full precision testing of the model

        Args:
            model: pytorch model
            num_classes: number of classes of the dataset
            data_loader: data loader of the testing dataset
            model_filename: string wit the directory in which the full-precision model will be stored
            best_accuracy: updated with the best accuracy (percentage) achieved during training
            save_model: bool that toggles model storage
        Returns:
            best_accuracy: best accuracy (percentage) achieved during training """
    print('===> Evaluate mode')

    # Switch to evaluate mode
    model.eval()

    loss = 0
    correct = 0

    num_batches = len(data_loader)

    for data, target in data_loader:
        batch_size = data.size(0)
        target_one_hot = one_hot_encode(target, length=num_classes)

        data, target_one_hot = Variable(data, volatile=True), Variable(target_one_hot)

        if torch.cuda.device_count() > 0:  # if there are available GPUs, move data to the first visible
            device = torch.device("cuda:0")
            data = data.to(device)
            target = target.to(device)
            target_one_hot = target_one_hot.to(device)

        # Output predictions
        output = model(data)  # output from DigitCaps (out_digit_caps)

        # Sum up batch loss

        m_loss = \
            capsnet_testing_loss(output, target_one_hot)
        loss += m_loss.data

        # Count number of correct predictions
        # Compute the norm of the vector capsules
        v_length = torch.sqrt((output ** 2).sum(dim=2))
        assert v_length.size() == torch.Size([batch_size, num_classes])

        # Find the index of the longest vector
        _, max_index = v_length.max(dim=1)
        assert max_index.size() == torch.Size([batch_size])

        # vector with 1 where the model makes a correct prediction, 0 where false
        correct_pred = torch.eq(target.cpu(), max_index.data.cpu())
        correct += correct_pred.sum()

    # Log test losses
    loss /= num_batches

    # Log test accuracies
    num_test_data = len(data_loader.dataset)
    accuracy_percentage = float(correct) * 100.0 / float(num_test_data)

    # Print test losses and accuracy
    print('Test: [Loss: {:.6f}'.format(
            loss))
    print('Test Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, num_test_data, accuracy_percentage))

    if accuracy_percentage > best_accuracy:
        best_accuracy = accuracy_percentage
        if save_model:
            torch.save(model.state_dict(), model_filename)

    return best_accuracy
