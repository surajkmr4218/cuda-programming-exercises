import time
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from PA2 import EX_CNN, training_data

if __name__ == "__main__":
    for bs in [32, 64, 128, 256, 512, 1024]:
        loader = DataLoader(training_data, batch_size=bs, shuffle=True, num_workers=4)
        model = EX_CNN()
        trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=1,
                             limit_train_batches=50, logger=False,
                             enable_checkpointing=False)
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        try:
            trainer.fit(model, train_dataloaders=loader)
            print(f"bs={bs}: {time.time()-start:.1f}s, "
                  f"peak mem {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        except torch.cuda.OutOfMemoryError:
            print(f"bs={bs}: OUT OF MEMORY")
            torch.cuda.empty_cache()