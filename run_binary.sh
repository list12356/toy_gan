python3 brs_gan.py --dir="mnist_binary_alpha_1000_loss1_sigma" --alpha=1000 --l=1 --mode="binary" --sigma=1&
python3 brs_gan.py --dir="mnist_binary_alpha_100_loss1_sigma" --alpha=100 --l=1 --mode="binary" --sigma=1&
python3 brs_gan.py --dir="mnist_binary_alpha_10_loss1_sigma" --alpha=10 --l=1 --mode="binary" --sigma=1&
python3 brs_gan.py --dir="mnist_binary_alpha_1_loss1_sigma" --alpha=1 --l=1 --mode="binary" --sigma=1&
python3 brs_gan.py --dir="mnist_binary_alpha_1000_loss1_nosigma" --alpha=1000 --l=1 --mode="binary" --sigma=0&
python3 brs_gan.py --dir="mnist_binary_alpha_100_loss1_nosigma" --alpha=100 --l=1 --mode="binary" --sigma=0&
python3 brs_gan.py --dir="mnist_binary_alpha_10_loss1_nosigma" --alpha=10 --l=1 --mode="binary" --sigma=0&
python3 brs_gan.py --dir="mnist_binary_alpha_1_loss1_nosigma" --alpha=1 --l=1 --mode="binary" --sigma=0&