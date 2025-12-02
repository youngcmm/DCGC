import json
import gc
from torch import optim
import opt
from setup import setup_args
from model import simple_unaugmented_graph_encoding
# import argparse
from opt import args
from utils_dcl import *
from kmeans_gpu import kmeans
import lightly.loss as lightLoss
from torch.nn.parallel import DataParallel

# if __name__ == '__main__':
# args = opt.args
dataset_name = args.dataset
setup_args(dataset_name)
# ten runs with different random seeds
# for args.seed in range(args.runs):
this_seed = 0
for seed in [random.randint(1, 307) for _ in range(3)]:
    # record results
    # fix the random seed
    setup_seed(seed)
    this_seed = seed
    # load graph data
    X, y, A, node_num, cluster_num = load_graph_data(dataset_name, show_details=False)

    # apply the laplacian filtering
    X_filtered = laplacian_filtering(A, X, args.t)

    # build our hard sample aware network
    SUGE = simple_unaugmented_graph_encoding(input_dim=X.shape[1], hidden_dim=args.dims, act=args.activate, n_num=node_num)
    SUGE = DataParallel(SUGE)

    # positive and negative sample pair index matrix
    mask = torch.ones([node_num * 2, node_num * 2]) - torch.eye(node_num * 2)

    # load data to device
    A, SUGE, X_filtered, mask = map(lambda x: x.to(args.device), (A, SUGE, X_filtered, mask))

    # test
    Z1, Z2, _, _ = SUGE(X_filtered, A)
    acc, nmi, ari, f1, y_hat, _ = cluster(X_filtered, y, cluster_num)

    # Initialize center1 and center2
    # center1 = torch.zeros(cluster_num, Z1.size(1)).to(args.device)
    # center2 = torch.zeros(cluster_num, Z2.size(1)).to(args.device)

    # adam optimizer
    optimizer = optim.Adam(SUGE.parameters(), lr=args.lr)

    # training
    # for epoch in tqdm(range(400), desc="training..."):
    for epoch in range(400):
        # print(epoch)
        # train mode
        SUGE.train()

        # encoding with Eq. (3)-(5)
        Z1, Z2, E1, E2 = SUGE(X_filtered, A)
        # if epoch % 10 == 0:

        if epoch > 200:
            index = torch.tensor(range(X_filtered.shape[0]), device=args.device)
            y_sam = torch.tensor(y_hat, device=args.device)
            index = index[torch.argsort(y_sam)]
            class_num = {}
            for label in torch.sort(y_sam).values:
                label = label.item()
                if label in class_num.keys():
                    class_num[label] += 1
                else:
                    class_num[label] = 1
            key = sorted(class_num.keys())
            centers_1 = torch.tensor([], device=args.device)
            centers_2 = torch.tensor([], device=args.device)
            for i in range(len(key[:-1])):
                class_num[key[i + 1]] = class_num[key[i]] + class_num[key[i + 1]]
                now = index[class_num[key[i]]:class_num[key[i + 1]]]
                centers_1 = torch.cat([centers_1, torch.mean(Z1, dim=0).unsqueeze(0)], dim=0)
                centers_2 = torch.cat([centers_2, torch.mean(Z2, dim=0).unsqueeze(0)], dim=0)

            centers_1 = F.normalize(centers_1, dim=1, p=2)
            centers_2 = F.normalize(centers_2, dim=1, p=2)
            S = comprehensive_similarity(Z1, Z2, E1, E2, SUGE.module.alpha)
            loss = dclInstance(S, mask, node_num) + dclCluster_v1(centers_1, centers_2)
        else:
            S = comprehensive_similarity(Z1, Z2, E1, E2, SUGE.module.alpha)
            loss = dclInstance(S, mask, node_num)

        # optimization
        loss.backward()
        optimizer.step()

        # testing and update weights of sample pairs
        if epoch % 10 == 0:
            # evaluation mode
            SUGE.eval()
            # encoding
            Z1, Z2, E1, E2 = SUGE(X_filtered, A)
            # fusion and testing
            Z = (Z1 + Z2) / 2
            acc, nmi, ari, f1, y_hat, _= cluster(Z, y, cluster_num)
            # print(acc, nmi, ari)

            # recording
            if acc >= args.acc:
                args.acc, args.nmi, args.ari, args.f1 = acc, nmi, ari, f1
                print(acc, nmi, ari, f1)
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    gc.collect()
params = {
    'acc': args.acc,
    'nmi': args.nmi,
    'ari': args.ari,
    'f1': args.f1,
    'datasetname': dataset_name,
    'seed' : this_seed,
}
with open('params.json', 'a') as f:
    json.dump(params, f)
f.close()

