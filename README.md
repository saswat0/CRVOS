# CRVOS
PyTorch implementation of CRVOS: Clue Refining Network For Video Object Segmentation

[Paper Link](https://arxiv.org/pdf/2002.03651.pdf)

### Download
* Dataset [DAVIS17](https://davischallenge.org/davis2017/code.html)
* Pre-trained [model](https://drive.google.com/a/yonsei.ac.kr/file/d/19DFvKs5N8wuRzjBKNg72szyGcyvFT0x6/view?usp=sharing)

### Usage
* Check fps
  ```bash
  python3 evaluate.py --fps
  ```
* Save outputs
  ```bash
  python3 evaluate.py --save
  ```

### Evaluation
* [Official evaluation code](https://github.com/davisvideochallenge/davis2017-evaluation)
* [Author's evaluation code](https://github.com/suhwan-cho/davis-evaluation)

### References:
* Author's page [Link](https://github.com/suhwan-cho/)
* [A-GAME](https://github.com/joakimjohnander/agame-vos)

### Citation
```
@article{cho2020crvos,
  title={CRVOS: Clue Refining Network for Video Object Segmentation},
  author={Cho, Suhwan and Cho, MyeongAh and Chung, Tae-young and Lee, Heansung and Lee, Sangyoun},
  journal={arXiv preprint arXiv:2002.03651},
  year={2020}
}
```