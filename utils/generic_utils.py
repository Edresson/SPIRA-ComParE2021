import os
import torch
import json
import random
from random import getrandbits
import re
import torch.nn.functional as F
import torch.nn as nn
from typing import Tuple

class F1Score:
    """
    Class for f1 calculation in Pytorch.
    Credits: https://stackoverflow.com/questions/62265351/measuring-f1-score-for-multiclass-classification-natively-in-pytorch
    """

    def __init__(self, average: str = 'weighted'):
        """
        Init.

        Args:
            average: averaging method
        """
        self.average = average
        if average not in [None, 'micro', 'macro', 'weighted']:
            raise ValueError('Wrong value of average parameter')

    @staticmethod
    def calc_f1_micro(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate f1 micro.

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels

        Returns:
            f1 score
        """
        true_positive = torch.eq(labels, predictions).sum().float()
        f1_score = torch.div(true_positive, len(labels))
        return f1_score

    @staticmethod
    def calc_f1_count_for_label(predictions: torch.Tensor,
                                labels: torch.Tensor, label_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate f1 and true count for the label

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels
            label_id: id of current label

        Returns:
            f1 score and true count for label
        """
        # label count
        true_count = torch.eq(labels, label_id).sum()

        # true positives: labels equal to prediction and to label_id
        true_positive = torch.logical_and(torch.eq(labels, predictions),
                                          torch.eq(labels, label_id)).sum().float()
        # precision for label
        precision = torch.div(true_positive, torch.eq(predictions, label_id).sum().float())
        # replace nan values with 0
        precision = torch.where(torch.isnan(precision),
                                torch.zeros_like(precision).type_as(true_positive),
                                precision)

        # recall for label
        recall = torch.div(true_positive, true_count)
        # f1
        f1 = 2 * precision * recall / (precision + recall)
        # replace nan values with 0
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1).type_as(true_positive), f1)
        return f1, true_count

    def __call__(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate f1 score based on averaging method defined in init.

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels

        Returns:
            f1 score
        """

        # simpler calculation for micro
        if self.average == 'micro':
            return self.calc_f1_micro(predictions, labels)

        f1_score = 0
        for label_id in range(1, len(labels.unique()) + 1):
            f1, true_count = self.calc_f1_count_for_label(predictions, labels, label_id)

            if self.average == 'weighted':
                f1_score += f1 * true_count
            elif self.average == 'macro':
                f1_score += f1

        if self.average == 'weighted':
            f1_score = torch.div(f1_score, len(labels))
        elif self.average == 'macro':
            f1_score = torch.div(f1_score, len(labels.unique()))

        return f1_score

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_config(config_path):
    config = AttrDict()
    with open(config_path, "r") as f:
        input_str = f.read()
    input_str = re.sub(r'\\\n', '', input_str)
    input_str = re.sub(r'//.*\n', '\n', input_str)
    data = json.loads(input_str)
    config.update(data)
    return config

def load_config_from_str(input_str):
    config = AttrDict()
    input_str = re.sub(r'\\\n', '', input_str)
    input_str = re.sub(r'//.*\n', '\n', input_str)
    data = yaml.load(input_str, Loader=yaml.FullLoader)
    config.update(data)
    return config

def copy_config_file(config_file, out_path, new_fields):
    config_lines = open(config_file, "r").readlines()
    # add extra information fields
    for key, value in new_fields.items():
        if isinstance(value, str):
            new_line = '"{}":"{}",\n'.format(key, value)
        else:
            new_line = '"{}":{},\n'.format(key, value)
        config_lines.insert(1, new_line)
    config_out_file = open(out_path, "w")
    config_out_file.writelines(config_lines)
    config_out_file.close()

def save_config_file(config, out_path):
    with open(out_path, 'w') as fp:
        json.dump(config, fp)


# adapted from https://github.com/digantamisra98/Mish/blob/master/Mish/Torch/mish.py
class Mish(nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, inp):
        '''
        Forward pass of the function.
        '''
        return inp * torch.tanh(F.softplus(inp))

def set_init_dict(model_dict, checkpoint, c):
    """
    This Function is adpted from: https://github.com/mozilla/TTS
    Credits: Eren Gölge (@erogol)
    """
    # Partial initialization: if there is a mismatch with new and old layer, it is skipped.
    for k, v in checkpoint['model'].items():
        if k not in model_dict:
            print(" | > Layer missing in the model definition: {}".format(k))
    # 1. filter out unnecessary keys
    pretrained_dict = {
        k: v
        for k, v in checkpoint['model'].items() if k in model_dict
    }
    # 2. filter out different size layers
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if v.numel() == model_dict[k].numel()
    }
    # 3. skip reinit layers
    if c.train_config.reinit_layers is not None:
        for reinit_layer_name in c.train_config.reinit_layers:
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if reinit_layer_name not in k
            }
    # 4. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    print(" | > {} / {} layers are restored.".format(len(pretrained_dict),
                                                     len(model_dict)))
    return model_dict


# https://github.com/mozilla/TTS/blob/ff295c65242328a6bc23a9fd9b4e6d819342795a/utils/training.py
# pylint: disable=protected-access
class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps=0.1, last_epoch=-1):
        self.warmup_steps = float(warmup_steps)
        super(NoamLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 1)
        return [
            base_lr * self.warmup_steps**0.5 *
            min(step * self.warmup_steps**-1.5, step**-0.5)
            for base_lr in self.base_lrs
        ]

def binary_acc(y_pred, y):
    """Calculates model accuracy
    
    Arguments:
        y_pred {torch.Tensor} -- Output of model between 0 and 1
        y {torch.Tensor} -- labels/target values
    
    Returns:
        [torch.Tensor] -- accuracy
    """
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag == y).float().sum()
    n = y.nelement() 
    acc = correct_results_sum/n
    acc = acc * 100
    return acc.item()


def save_best_checkpoint(log_dir, model, optimizer, c, step, val_loss, best_loss, early_epochs=None):
    if val_loss < best_loss:
        best_loss = val_loss
        if early_epochs is not None:
            early_epochs = 0
        save_path = os.path.join(log_dir, 'best_checkpoint.pt')
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
            'config_str': str(c),
        }, save_path)
        print("\n > BEST MODEL ({0:.5f}) : {1:}".format(
            val_loss, save_path))
    else:
        if early_epochs is not None:
            early_epochs += 1
    return best_loss, early_epochs