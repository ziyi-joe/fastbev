## 一、环境<br />
   * conda create -n fastbev python=3.8<br />
   * conda activate fastbev<br />
   * conda activate [name of your env]<br />
   * pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html<br />
   * pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio -f https://download.pytorch.org/whl/torch_stable.html<br />
   * pip install mmcv-full==1.4.0 mmdet==2.14.0 mmsegmentation==0.14.1 ipdb timm<br />
   * python setup.py develop<br />
   * 根据工程需要安装缺少的基础包，相关版本可参考fastbev_env.txt<br />
   * 所用数据集为开源nuscenes数据集<br />

## 二、train<br />
训练好的pth权重保存在ckpts/epoch15_batch4_lr0002.pth
```bash
bash tools/run_train.sh
```

## 三、test<br />
修改"configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py"配置中test_mode='test_pth'
```bash
bash tools/run_test.sh
```

## 四、导出onnx模型<br />
```python
python ./tools/export_onnx.py --config ./configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py --checkpoint ./ckpts/epoch15_batch4_lr0002.pth
```

## 五、测试onnx模型精度<br />
修改"configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py"配置中test_mode='test_onnx'，并正确配置所使用onnx模型的路径
```bash
bash tools/run_test.sh
```

## 六、compiler编译<br />
### generate calibrate data
```python
python tools/generate_calibrate_data.py
```

### build

```
cd ./compiler/
bstnnx_run --config fastbev_backbone_c1200.yaml --result_dir ./fastbev_backbone_result_c1200/
bstnnx_run --config fastbev_head_c1200.yaml --result_dir ./fastbev_head_result_c1200/ --extra priority_range=100-250
```
按照如上流程编译完了之后，需要手动修改head部分的输入scale，使其和backbone输出scale一致，从而提高板端部署效率，具体操作如下：<br />
修改 ./fastbev_head_result_c1200/250_ModelSolidationStage/calibrate_param_prepare.json，将json文件中 <br />
```
"0": fixed,
"1": fixed,
"2": fixed,
"3": fixed,
```
修改为
```
"0": fixed_-8_8,
"1": fixed_-8_8,
"2": fixed_-8_8,
"3": fixed_-8_8,
```
然后执行
```
bstnnx_run --config fastbev_head_c1200.yaml --result_dir ./fastbev_head_result_c1200/ --extra resume_from_breakpoint=true
```
需要注意的是，fixed_-X_X这里X的值由backbone部分输出的范围决定，开源fastbev中backbone部分输出数据分布在±8之间，因此这里填fixed_-8_8


## 七、结果分析<br />
| Model    | A1000 FPS | C1200 FPS | MAP (Float) | MAP (Quant) |
|----------|--------|-------------|-------------|-------------|
| origin  | --  | --       | 27.68      | --       |
| bst_overall  | --  | --       | 25.9      | 25.37       |
| bst_backbone  | 559  | 547       |  --     | --       |
| bst_head  | 94  | 112       | --      | --       |
| bst_postprocess(CPU)  | 7ms  |  7ms      | --      | --       |

- 投影部分在cpu端耗时约7ms（当前demo runtime仅使用engine0，4maca，后续用户可进一步使用双engine做pipeline优化）<br />

## 八、pc端与板端对比<br />
   * 部署端包括使用24张图 -> stage1 2d model -> stage2 mapping -> stage3 3d model三段的pipeline，用户可自行比较两者是否一致，板端输出已保存一份在bst_deploy/A1000B0_output文件夹下，具体参考tools/inference_compare.py代码119-121、179-181、198-205行<br />
   * 可视化部署端结果，将tools/vis_a1000_output.py中189-195行打开，运行python ./tools/vis_a1000_output.py即可，注：此脚本仅可可视化给出的24张图<br />
