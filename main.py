from diffusion.diffusion import *
from diffusion.diffusion_trainer import *
from diffusion.model_wrappers import *
from ffnn.model import *
from torch.utils.data import DataLoader


def main():
    data_path = 'data/elia_data'
    train_loader = torch.load(f'{data_path}/train_loader_non_spatial.pt')
    val_loader = torch.load(f'{data_path}/val_loader_non_spatial.pt')
    test_loader = torch.load(f'{data_path}/test_loader_non_spatial.pt')
    transform = torch.load(f'{data_path}/transform_non_spatial.pt')

    x = next(iter(train_loader))[0]
    y = next(iter(train_loader))[1].unsqueeze(-1)
    x_shape = x.shape
    y_shape = y.shape

    print(x_shape)
    print(y_shape)

    F = FFNN(input_size=96, num_hidden=2, hidden_size=128, output_size=96)
    out = F(y)
    print(out.shape)
    wrapper = DiffusionModel(
        model=F, data_pred=True, input_constructor=BaseConstructor(), output_shape=(96,))
    out = wrapper(x, y, 0)
    print(out.shape)

    base_loader = train_loader
    fp = ForwardProcess(torch.Tensor([0.1, 0.2, 0.9]))
    train_loader = DiffusionLoader(
        base_loader=train_loader, forward_process=fp)
    trainer = DiffusionTrainer(model=wrapper, forward_process=fp)

    for x, y in val_loader:
        print(x.shape, y.shape)

    trainer.train(num_epochs=50, diffusion_loader=train_loader,
                  val_loader=val_loader)


if __name__ == "__main__":

    main()
