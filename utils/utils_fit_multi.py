import torch
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
from tqdm import tqdm
from contextlib import nullcontext
from utils.utils import get_lr
from utils.utils_metrics import f_score
import torch.distributed as dist

def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val,
                  gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, args, scaler, warmup_scheduler, warmup_epoch):
    wz = int(dist.get_world_size())
    total_loss = 0
    total_f_score = 0
    val_loss = 0
    val_f_score = 0
    model_train.train()
    if dist.get_rank() == 0:
        print('Start Train')
        pbar = tqdm(total=int(epoch_step/wz), desc=f'Epoch {(epoch + 1)}/{Epoch}', postfix=dict, mininterval=0.3)
    else:
        pbar = None
    #with torch.cuda.amp.autocast(enabled=scaler is not None):
    for iteration, batch in enumerate(gen):
        if epoch < warmup_epoch:
            warmup_scheduler.step()
            warm_lr = warmup_scheduler.get_lr()[0]
        if iteration >= epoch_step:
            break
        imgs, pngs, labels = batch

        with torch.no_grad():
            imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
            pngs = torch.from_numpy(pngs).long()
            labels = torch.from_numpy(labels).type(torch.FloatTensor)
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.to(args.local_rank)
                pngs = pngs.to(args.local_rank)
                labels = labels.to(args.local_rank)
                weights = weights.to(args.local_rank)

        if args.accumulation_steps is not None:
            my_context = model.no_sync if args.local_rank != -1 and iteration % args.accumulation_steps != 0 else nullcontext
            with my_context():
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    outputs = model_train(imgs)
                    #outputs = outputs.float()
                    if focal_loss:
                        loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                    else:
                        loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                    if dice_loss:
                        main_dice = Dice_loss(outputs, labels)
                        loss = loss + main_dice

                with torch.no_grad():
                    # -------------------------------#
                    #   计算f_score
                    # -------------------------------#
                    _f_score = f_score(outputs, labels)

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                #loss.backward()

                if iteration % args.accumulation_steps == 0:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    else:
                        optimizer.step()
                        optimizer.zero_grad()

                total_loss += loss.item()
                total_f_score += _f_score.item()

        else:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model_train(imgs)
                #outputs = outputs.float()
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice

            with torch.no_grad():
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                _f_score = f_score(outputs, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            # loss.backward()
            # optimizer.step()
            total_loss += loss.item()
            total_f_score += _f_score.item()

        if dist.get_rank() == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': warm_lr if epoch<warmup_epoch else get_lr(optimizer) })
            pbar.update(1)


    model_train.eval()
    if dist.get_rank() == 0:
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=int(epoch_step_val/wz), desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    else:
        pbar = None
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
            pngs = torch.from_numpy(pngs).long()
            labels = torch.from_numpy(labels).type(torch.FloatTensor)
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.to(args.local_rank)
                pngs = pngs.to(args.local_rank)
                labels = labels.to(args.local_rank)
                weights = weights.to(args.local_rank)

            outputs = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice
            # -------------------------------#
            #   计算f_score
            # -------------------------------#
            _f_score = f_score(outputs, labels)

            val_loss += loss.item()
            val_f_score += _f_score.item()

        if dist.get_rank() == 0:
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                'f_score': val_f_score / (iteration + 1),
                                'lr': warm_lr if epoch<warmup_epoch else get_lr(optimizer)})
            pbar.update(1)

    loss_history.append_loss(total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1))
    if dist.get_rank() == 0:
        print('Finish Validation')
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1)))
        torch.save(model.module.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (
        (epoch + 1), total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1)))


def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda, dice_loss,
                         focal_loss, cls_weights, num_classes):
    total_loss = 0
    total_f_score = 0

    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            imgs, pngs, labels = batch

            with torch.no_grad():
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs = torch.from_numpy(pngs).long()
                labels = torch.from_numpy(labels).type(torch.FloatTensor)
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()
                    weights = weights.cuda()

            optimizer.zero_grad()

            outputs = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            with torch.no_grad():
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_f_score += _f_score.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    loss_history.append_loss(total_loss / (epoch_step + 1))
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f' % (total_loss / (epoch_step + 1)))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f.pth' % ((epoch + 1), total_loss / (epoch_step + 1)))
