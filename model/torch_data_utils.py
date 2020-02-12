import torch

class QADataset(torch.utils.data.Dataset):
    def __init__(self, ids, masks, segments, labels=None):
        self.ids = ids
        self.masks = masks
        self.segments = segments
        self.labels = labels

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.ids[idx], self.masks[idx], self.segments[idx], self.labels[idx]
        else:
            return self.ids[idx], self.masks[idx], self.segments[idx]

    def __len__(self):
        return len(self.ids)

class QADataset_SeparateQA(torch.utils.data.Dataset):
    def __init__(self, q_ids, q_masks, q_segments, a_ids, a_masks, a_segments, labels=None):
        self.q_ids = q_ids
        self.q_masks = q_masks
        self.q_segments = q_segments
        self.a_ids = a_ids
        self.a_masks = a_masks
        self.a_segments = a_segments
        self.labels = labels

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.q_ids[idx], self.q_masks[idx], self.q_segments[idx], self.a_ids[idx], self.a_masks[idx], self.a_segments[idx], self.labels[idx]
        else:
            return self.q_ids[idx], self.q_masks[idx], self.q_segments[idx]

    def __len__(self):
        return len(self.q_ids)

def get_dataloader(dataset, batch_size, shuffle=True, weights=None):
    if weights is not None:
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(dataset), replacement=True)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


