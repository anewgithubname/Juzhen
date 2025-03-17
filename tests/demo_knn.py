import time
import torch
import torchvision
import torchvision.transforms as transforms

device = 'cpu'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_mnist_dataset(train_size=2000, test_size=200):
    """
    Loads MNIST via torchvision, returns smaller subsets for demonstration.
    Shapes:
       train_data: [train_size, 784], train_labels: [train_size]
       test_data:  [test_size, 784],  test_labels:  [test_size]
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert [0..1], shape [1,28,28]
    ])

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Slice subsets
    train_data = train_dataset.data[:train_size].reshape(train_size, -1).float() / 255.0
    train_labels = train_dataset.targets[:train_size]
    test_data = test_dataset.data[:test_size].reshape(test_size, -1).float() / 255.0
    test_labels = test_dataset.targets[:test_size]

    return train_data, train_labels, test_data, test_labels

def knn_predict_with_mm(test_data, train_data, train_labels, k=3):
    """
    KNN prediction for all test samples using the formula:
        dist_sq(x,y) = ||x||^2 + ||y||^2 - 2 x·y
    where x·y is computed via matrix multiplication.

    :param test_data:  shape [T, D]
    :param train_data: shape [N, D]
    :param train_labels: shape [N]
    :param k: number of neighbors
    :return: predictions shape [T]
    """

    # 1) Compute x^2 for each test sample (row-wise norm-squared)
    #    shape: [T, 1]
    x2 = (test_data ** 2).sum(dim=1, keepdim=True)

    # 2) Compute y^2 for each train sample (row-wise norm-squared)
    #    shape: [N]  (we will broadcast this)
    y2 = (train_data ** 2).sum(dim=1)

    # 3) Compute the dot products test_data @ train_data.T
    #    shape: [T, N]
    xy = test_data @ train_data.t()

    # 4) dist_sq = x2 + y2 - 2 * xy
    #    broadcasting: x2 is [T,1], y2 is [N], so final is [T, N]
    #    (T,1) + (N,) -> (T,N) due to broadcasting
    dist_sq = x2 + y2 - 2 * xy

    # 5) For each test sample, find indices of k smallest distances
    #    shape of 'indices': [T, k]
    _, indices = torch.topk(dist_sq, k, dim=1, largest=False)

    # 6) Gather the training labels of these neighbors
    #    shape: [T, k]
    knn_labels = train_labels[indices]

    # 7) Majority vote (most common label among the k neighbors)
    #    shape: [T]
    #    torch.mode(..., dim=1) returns (values, counts)
    predictions = torch.mode(knn_labels, dim=1).values

    return predictions

def evaluate_knn_with_mm(test_data, test_labels, train_data, train_labels, k=3):
    """
    Classify the entire test set via our matrix-multiplication-based KNN,
    then compute accuracy.
    """
    preds = knn_predict_with_mm(test_data, train_data, train_labels, k=k)
    accuracy = (preds == test_labels).float().mean() * 100
    return accuracy.item()

def main():
    # 1) Load dataset subsets
    train_size = 60000
    test_size = 10000
    train_data, train_labels, test_data, test_labels = load_mnist_dataset(
        train_size, test_size
    )
    print("Train data shape:", train_data.shape)
    print("Test data shape: ", test_data.shape)

    train_data = train_data.to(device)
    train_labels = train_labels.to(device)
    test_data = test_data.to(device)
    test_labels = test_labels.to(device)

    # 2) KNN classification with matrix-multiplication distance
    k = 7
    print(f"\nRunning KNN with k={k} on {test_size} test samples...")
    starttime = time.time()
    accuracy = evaluate_knn_with_mm(test_data, test_labels, train_data, train_labels, k=k)
    print(f"Time taken: {time.time() - starttime:.2f} sec")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
