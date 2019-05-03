### Run the porgram

```python
python main.py --arch=conv_net --epochs=100
```
```python
python main.py --arch=resnet18 --data_transforms=resnet_input_transform --epochs=100
```
```python
python main.py --arch=cifar_convnet --dataset=cifar10 --data_transforms=cifar10_input_transform
```
```python
python main.py --arch=conv_net --epochs=100
```

python main.py --arch=cifar_convnet --dataset=cifar10 --data_transforms=cifar10_input_transform --algorithm=self_training --epoch=10 --save_model=y --save_interval=2 --proxy_interval=2 --tau=0.5


python main.py --arch=cifar_convnet --dataset=cifar10 --data_transforms=cifar10_input_transform --algorithm=self_training --epoch=10 --save_model=y --save_interval=2 --proxy_interval=2 --tau=0.5

python main.py --arch=resnet18 --dataset=ssl_data --data_transforms=resnet_input_transform --algorithm=self_training --epochs=20 --proxy_interval=2 --tau=0.7


python main.py --arch=resnet18 --dataset=ssl_data --data_transforms=resnet_input_transform --algorithm=just_supervised --epochs=200 --save_model=y --save_interval=20