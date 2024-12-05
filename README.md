# CO-DETR的TensorRT部署

> 参考  
https://github.com/DataXujing/Co-DETR-TensorRT  
https://github.com/Sense-X/Co-DETR/issues/26

## 环境版本
> ubuntu 22.04下基于2080ti和v100均成功部署  
python == 3.8.20  
tensorrt == 8.5.3.1  
cuda == 11.8  
cudnn == 8.6.0.163  
mmdeploy == 1.3.1  
mmdet == 3.3.0  
mmcv == 2.1.0  
mmengine == 0.10.4  
polygraphy == 0.49.9  
onnxruntime == 1.19.2  
onnxsim == 0.4.36

## 部署流程

### Pytorch模型转ONNX
1.修改mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_1x_coco.py  

```
test_cfg=[  
          # # Deferent from the DINO, we use the NMS.  
            dict(
                max_per_img=300,
                # NMS can improve the mAP by 0.2.
                # nms=dict(type='soft_nms', iou_threshold=0.8)),  # 关掉test过程中的soft nms
        ),
```

2.修改mmdeploy关于onnx的导出配置  

```
# mmdeploy/configs/_base_/onnx_config.py
onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,  # opset 版本
    save_file='end2end.onnx',  #转出onnx的保存名字
    input_names=['input'],  # input的名字
    output_names=['output'],  # output的名字
    input_shape=None,
    optimize=True)
# mmdeploy/configs/mmdet/_base_/base_static.py

_base_ = ['../../_base_/onnx_config.py']

onnx_config = dict(output_names=['dets', 'labels'], input_shape=[640,640])  # static input的大小设置为640x640，设置为None即多尺寸适应
codebase_config = dict(
    type='mmdet',
    task='ObjectDetection',
    model_type='end2end',
    post_processing=dict(
        score_threshold=0.05,
        confidence_threshold=0.005,  # for YOLOv3
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=5000,
        keep_top_k=100,
        background_label_id=-1,
    ))

# co-dino使用了多尺度训练，这里我们将test input的尺度设为640x640,减少计算量
```

3.模型输出结构改动

```
# 修改mmdetection/projects/CO-DETR/codetr/co_dino_head.py的_predict_by_feat_single函数
原代码:
        if score_thr > 0:
            valid_mask = scores > score_thr
            scores = scores[valid_mask]
            bbox_pred = bbox_pred[valid_mask]
            det_labels = det_labels[valid_mask]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            assert img_meta.get('scale_factor') is not None
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))

        results = InstanceData()
        # det_bboxes = bbox_pred
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels

修改为:
        # 把模型结构里的xywh -> xyxy函数脱离
        # if score_thr > 0:
        #     valid_mask = scores > score_thr
        #     scores = scores[valid_mask]
        #     bbox_pred = bbox_pred[valid_mask]
        #     det_labels = det_labels[valid_mask]
        # 
        # det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        # det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        # det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        # det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        # det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        # if rescale:
        #     assert img_meta.get('scale_factor') is not None
        #     det_bboxes /= det_bboxes.new_tensor(
        #         img_meta['scale_factor']).repeat((1, 2))

        results = InstanceData()
        det_bboxes = bbox_pred
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels

```

4.mmdeploy转onnx

```
python mmdeploy/tools/deploy.py \
        mmdeploy/configs/mmdet/detection/detection_onnxruntime_static.py \
        mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py \
        mmdetection/checkpoints/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth \
        mmdetection/demo/demo.jpg \
        --work-dir mmdetection/checkpoints \
        --device cpu
# 这个过程生成了end2end.onnx的，但是onnxruntime的时候或报错，报错的原因是grid_sampler算子onnxruntime和tensorrt均不支持，稍后会编译tensorrt plugin解决该伪问题
```

5.对onnx进行fold constants，进行onnxsim会导致漏检，可选择性放弃

```
polygraphy surgeon sanitize end2end.onnx --fold-constants -o end2end_folded.onnx
python -m onnxsim end2end_folded.onnx end2end_folded_sim.onnx
```

### ONNX模型转TensorRT Engine模型
把**end2end_folded.onnx**模型转为TensorRT engine模型，采用fp16或best参数也会导致漏检，可选择性放弃，转换参考命令:

```
# 需先配置运行trtexec的链接库路径，包括cuda，cudnn和TensorRT的lib路径

trtexec --onnx=path_to_end2end_folded.onnx --saveEngine=CO-DETR_folded_fp32.engine --plugins=path_to_libmmdeploy_tensorrt_ops.so_within_installed_mmdeploy_package
```

### CO-DETR的TensorRT Engine模型推理
> 参考**inference_tensorrt.py**文件，一般步骤如下:  
> 1. 读取图片，进行letterbox处理和归一化；  
> 2. 初始化TensorRT引擎，加载TensorRT的插件；  
> 3. 运行TensorRT engine模型推理；  
> 4. xywh -> xyxy；  
> 5. agnostic的nms后处理。
