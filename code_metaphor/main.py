import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, BCELoss

from data_utils import get_data_list, data_preprocess, get_data_loader, calc_metrics, calc_metrics_binary
from model import MultimodalModel

import argparse

import warnings
warnings.filterwarnings('ignore')



def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_model', default='bert', help='Specify the text model')
    parser.add_argument('--image_model', default='vit', help='Specify the image model')
    parser.add_argument('--path', default='/data', help='Specify the dataset path')
    parser.add_argument('--language', default='zh', help='Set language: zh/en')
    parser.add_argument('--text_only', action='store_true', help='Use text only')
    parser.add_argument('--image_only', action='store_true', help='Use image only')
    parser.add_argument('--text_image', action='store_true', help='Use both text and image')
    parser.add_argument('--do_test', action='store_true', help='Run inference on the test set using the trained model')
    parser.add_argument('--lr', default=3e-5, help='Set learning rate', type=float)
    parser.add_argument('--weight_decay', default=1e-3, help='Set weight decay', type=float)
    parser.add_argument('--epochs', default=10, help='Set number of training epochs', type=int)
    parser.add_argument('--seed', default=20, help='Set random seed', type=int)
    args = parser.parse_args()
    return args


args = init_argparse()

"""text_only和image_only"""
assert((args.text_only and args.image_only) == False)

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

torch.backends.cudnn.deterministic = True


def model_train():

    resdir = './result'
    devResdir = './dev_result'
    if not os.path.exists(resdir):
        os.makedirs(resdir)

    if not os.path.exists(devResdir):
        os.makedirs(devResdir)

    train_data_list, val_data_list, test_data_list = get_data_list(args.path, args.language)
    train_data_list, val_data_list, test_data_list = data_preprocess(train_data_list, val_data_list, test_data_list)
    train_data_loader, valid_data_loader, test_data_loader = get_data_loader(train_data_list, val_data_list, test_data_list)

    # model = MultimodalModel.from_pretrained(args.text_model, args.image_model)
    model = MultimodalModel(args.text_model, args.image_model)
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(lr=args.lr, params=optimizer_grouped_parameters)

    criterion = BCELoss()
    best_rate = 0

    print('[START_OF_TRAINING_STAGE]')
    for epoch in range(args.epochs):
        total_loss = 0
        correct = 0
        total = 0
        target_list = []
        pred_list = []
        model.train()
        for idx, (guid, tag, image, text, senti) in enumerate(train_data_loader):
            # tag_three = F.one_hot(tag, num_classes=3).float().to(device)
            tag = tag.to(device)
            image = image.to(device)
            text = text.to(device)
            senti = senti.to(device)

            if args.text_only:
                out = model(image_input=None, text_input=text, senti_input=None)
            elif args.image_only:
                out = model(image_input=image, text_input=None, senti_input=None)
            else:
                out = model(image_input=image, text_input=text, senti_input=senti)


            loss = criterion(out, tag)
            loss.backward()
            torch.cuda.empty_cache()
            optimizer.step()
            optimizer.zero_grad()



            total_loss += loss.item() * len(guid)

            pred = torch.round(out).int()
            total += len(guid)
            correct += (pred == tag).sum()

            target_list.extend(tag.cpu().tolist())
            pred_list.extend(pred.cpu().tolist())

        total_loss /= total
        print('[EPOCH{:02d}]'.format(epoch + 1), end='')
        print('[TRAIN] - LOSS:{:.6f}'.format(total_loss), end='')
        rate = correct / total * 100
        print(' ACC_RATE:{:.2f}%'.format(rate), end='')
        metrics = calc_metrics(target_list, pred_list)
        print(' ACC: {:.2f}% PRE_W:{:.2f}% REC_w: {:.2f}% F1_w: {:.2f}% PRE_M:{:.2f}% REC_M: {:.2f}% F1_M: {:.2f}%' .format(metrics[0] * 100,
                                                                           metrics[1] * 100,
                                                                           metrics[2] * 100,
                                                                           metrics[3] * 100,
                                                                           metrics[4] * 100,
                                                                           metrics[5] * 100,
                                                                           metrics[6] * 100))

        total_loss = 0
        correct = 0
        total = 0
        target_list = []
        pred_list = []
        model.eval()

        dev_result = {}
        for guid, tag, image, text, senti in valid_data_loader:
            tag = tag.to(device)
            image = image.to(device)
            text = text.to(device)
            senti = senti.to(device)
            if args.text_only:
                out = model(image_input=None, text_input=text, senti_input=None)
            elif args.image_only:
                out = model(image_input=image, text_input=None, senti_input=None)
            else:
                out = model(image_input=image, text_input=text, senti_input=senti)

            loss = criterion(out, tag)

            total_loss += loss.item() * len(guid)
            pred = torch.round(out).int()
            total += len(guid)
            correct += (pred == tag).sum()

            target_list.extend(tag.cpu().tolist())
            pred_list.extend(pred.cpu().tolist())

            for i in range(len(guid)):
                dev_result[str(guid[i])] = [int(tag[i].item()), pred[i].item()]


        with open(os.path.join(devResdir, "result_" + str(epoch) + ".json"), "w") as f:
            json.dump(dev_result, f)

        total_loss /= total
        print('         [EVAL]  - LOSS:{:.6f}'.format(total_loss), end='')
        rate = correct / total * 100
        print(correct)
        # print(' ACC_RATE:{:.2f}%'.format(rate), end='')
        metrics = calc_metrics(target_list, pred_list)
        print(' ACC: {:.2f}% PRE_W:{:.2f}% REC_w: {:.2f}% F1_w: {:.2f}% PRE_M:{:.2f}% REC_M: {:.2f}% F1_M: {:.2f}%' .format(metrics[0] * 100,
                                                                           metrics[1] * 100,
                                                                           metrics[2] * 100,
                                                                           metrics[3] * 100,
                                                                           metrics[4] * 100,
                                                                           metrics[5] * 100,
                                                                           metrics[6] * 100))
        
        if rate > best_rate:
            best_rate = rate
            print('         [SAVE] BEST ACC_RATE ON THE VALIDATION SET:{:.2f}%'.format(rate))

            total_loss = 0
            correct = 0
            total = 0
            target_list = []
            pred_list = []
            model.eval()

            result_dict = {}
            for guid, tag, image, text, senti in test_data_loader:
                tag = tag.to(device)
                image = image.to(device)
                text = text.to(device)
                senti = senti.to(device)

                if args.text_only:
                    out = model(image_input=None, text_input=text, senti_input=None)
                elif args.image_only:
                    out = model(image_input=image, text_input=None, senti_input=None)
                else:
                    out = model(image_input=image, text_input=text, senti_input=senti)

                loss = criterion(out, tag)

                total_loss += loss.item() * len(guid)
                pred = torch.round(out).int()

                total += len(guid)
                correct += (pred == tag).sum()

                target_list.extend(tag.cpu().tolist())
                pred_list.extend(pred.cpu().tolist())

                for i in range(len(guid)):
                    result_dict[str(guid[i])] = [int(tag[i].item()), pred[i].item()]

                    # Output the incorrect predictions
                    if pred[i] != tag[i]:
                        print(f"Test,Incorrect Prediction: GUID: {guid[i]}, Pred: {pred[i].item()}, Tag: {tag[i].item()}")

            with open(os.path.join(resdir, "result_" + str(epoch) + ".json"), "w") as f:
                json.dump(result_dict, f)

            total_loss /= total
            print('         [TEST]  - LOSS:{:.6f}'.format(total_loss), end='')

            rate = correct / total * 100
            # print(' ACC_RATE:{:.2f}%'.format(rate), end='')
            metrics = calc_metrics(target_list, pred_list)
            print(' ACC: {:.2f}% PRE_W:{:.2f}% REC_w: {:.2f}% F1_w: {:.2f}% PRE_M:{:.2f}% REC_M: {:.2f}% F1_M: {:.2f}%' .format(metrics[0] * 100,
                                                                            metrics[1] * 100,
                                                                            metrics[2] * 100,
                                                                            metrics[3] * 100,
                                                                            metrics[4] * 100,
                                                                            metrics[5] * 100,
                                                                            metrics[6] * 100))
                # torch.save(model.state_dict(), 'model.pth')
            print()
    print('[END_OF_TRAINING_STAGE]')

if __name__ == "__main__":
    model_train()