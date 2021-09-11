from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import v_measure_score
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

@torch.no_grad()
def test(model, data):
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

def train(model, data, optimizer, nepoch):
    best_val_acc = test_acc = 0
    for epoch in range(1, nepoch):
        model.train()
        optimizer.zero_grad()
        F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
        optimizer.step()
        train_acc, val_acc, tmp_test_acc = test(model, data)
        best_val_acc = max(best_val_acc, val_acc)
        test_acc = max(test_acc, tmp_test_acc)
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, maxVal: {:.4f}, maxTest: {:.4f}'
        print(log.format(epoch, train_acc, val_acc, tmp_test_acc, best_val_acc, test_acc))

    model.eval()
    toraltest = torch.tensor([True for i in range(len(data.y))])
    pred = model()[toraltest].max(1)[1]
    ari = adjusted_rand_score(data.y.tolist(), pred.tolist())
    vm = v_measure_score(data.y.tolist(), pred.tolist())
    print("Adjusted Rand Index:", ari)
    print("V-measure:", vm)
    print("Confusion Matrix:", confusion_matrix(data.y.tolist(), pred.tolist()))
