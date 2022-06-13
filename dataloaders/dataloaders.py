def get_dataset_name(mode):
    if mode == "landscape":
        return "LandscapeDataset"
    if mode == "landscapetest":
        return "LandscapeTestDataset"
    else:
        return ValueError("There is no such dataset regime as %s" % mode)


def get_dataloaders(opt):
    dataset_name = get_dataset_name(opt.dataset_mode)
    file = __import__("dataloaders." + dataset_name)

    # test_only代表使用官方测试集
    if opt.phase == "test" and opt.test_only:
        # test
        dataloader_test = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=True)
        dataloader_test.set_attrs(batch_size=opt.batch_size, shuffle=False, drop_last=False, num_workers=4)

        print("[TEST] Created %s, size test: %d" % (dataset_name, dataloader_test.total_len))

        # 保持接口统一性。加上None后，在test.py中就不需要修改接收的返回值数量了
        return None, dataloader_test
    else:
        # train
        dataloader_train = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=False)
        dataloader_train.set_attrs(batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=4)
        # val
        dataloader_val = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=True)
        dataloader_val.set_attrs(batch_size=opt.batch_size, shuffle=False, drop_last=False, num_workers=4)

        print("[TRAIN] Created %s, size train: %d, size val: %d" % (
        dataset_name, dataloader_train.total_len, dataloader_val.total_len))

        return dataloader_train, dataloader_val
