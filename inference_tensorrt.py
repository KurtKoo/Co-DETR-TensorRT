# import os
import sys


# import numpy as np
# import tensorrt as trt
# import cv2
# from PIL import Image
sys.path.insert(1, "/home/rookie/Projects/TensorRT-8.5.3.1.Linux.x86_64-gnu.cuda-11.8.cudnn8.6/TensorRT-8.5.3.1/samples/python")
import common

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
# import pycuda.autoinit
try:
    import pycuda.autoprimaryctx
except ModuleNotFoundError:
    import pycuda.autoinit

import ctypes

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

engine_file = "/home/rookie/Projects/mmdetection/TensorRT-8.5/CO-DETR_folded_fp32.engine"
image_file = "/home/rookie/Projects/mmdetection/fall_2_866431070012109_20240909_145900_08d40445fe164b9db62de49d85b8f0a0_S56_D1800_H-462_Ah-19_Av-1_MudNone_MlrNone.jpg"
plugin_path = '/home/rookie/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmdeploy/lib/libmmdeploy_tensorrt_ops.so'

# 图像预处理函数
def letterbox(img, new_shape=(640, 640), color=(0, 0, 0), auto=True, scaleFill=False, scaleup=True):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh) # dw, left width padding. dh, top height padding.

def preprocess_image(image_path, input_shape=(640, 640)):
    image = cv2.imread(image_path)
    original_shape = image.shape[:2]  # (height, width)

    letterbox_image, r, pad = letterbox(image, auto=False, scaleFill=False, scaleup=False)
    cv2.imwrite("letterbox.jpg", letterbox_image)
    letterbox_original_image = letterbox_image

    letterbox_image = cv2.cvtColor(letterbox_image, cv2.COLOR_BGR2RGB)

    # 转换为浮点数并归一化
    letterbox_image = letterbox_image.astype(np.float32)
    letterbox_image -= np.array([123.675, 116.28, 103.53])
    letterbox_image /= np.array([58.395, 57.12, 57.375])

    # 转换为CHW格式并添加批次维度
    letterbox_image = letterbox_image.transpose((2, 0, 1))  # HWC to CHW
    letterbox_image = np.expand_dims(letterbox_image, axis=0)  # Add batch dimension

    return letterbox_image, r, pad, letterbox_original_image

def load_plugin(plugin_path, plugin_name):
    # 初始化 TensorRT 插件注册表
    trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), '')

    # 加载插件库
    ctypes.CDLL(plugin_path)

    # 获取插件注册表
    plugin_registry = trt.get_plugin_registry()

    # 查找插件创建器
    plugin_creator = next((c for c in plugin_registry.plugin_creator_list if c.name == plugin_name), None)
    if plugin_creator is None:
        raise RuntimeError(f'Failed to find the plugin creator for {plugin_name}')

    # 创建插件对象
    plugin_fields = trt.PluginFieldCollection([])
    plugin = plugin_creator.create_plugin(plugin_name, plugin_fields)
    if plugin is None:
        raise RuntimeError(f'Failed to create the plugin {plugin_name}')

    return plugin

# 创建执行上下文
def create_context(engine):
    return engine.create_execution_context()

def infer(engine, context, inputs, input_name, output_names):
    # 分配CUDA设备内存
    d_inputs = [cuda.mem_alloc(val.nbytes) for val in inputs.values()]
    d_outputs = [cuda.mem_alloc(val.nbytes) for val in inputs.values()]

    # 将数据从主机复制到设备
    for h_input, d_input in zip(inputs.values(), d_inputs):
        cuda.memcpy_htod(d_input, h_input)

    # 设置绑定
    bindings = [d_input.ptr for d_input in d_inputs] + [d_output.ptr for d_output in d_outputs]

    # 执行推理
    context.execute_v2(bindings)

    # 将结果从设备复制回主机
    outputs = {}
    for name, d_output in zip(output_names, d_outputs):
        output = np.empty_like(inputs[input_name])
        cuda.memcpy_dtoh(output, d_output)
        outputs[name] = output

    return outputs


def nms(boxes, scores, labels, iou_threshold):
    """
    非极大值抑制 (Non-Maximum Suppression, NMS)

    参数:
    boxes (numpy.ndarray): 形状为 (N, 4) 的数组，表示 N 个边界框，每个边界框的形式为 [x1, y1, x2, y2]。
    scores (numpy.ndarray): 形状为 (N,) 的数组，表示每个边界框的得分。
    labels (numpy.ndarray): 形状为 (N,) 的数组，表示每个边界框的类别标签。
    iou_threshold (float): 交并比 (IoU) 的阈值，用于判断边界框是否重叠。

    返回:
    numpy.ndarray: 保留的边界框的索引。
    """
    # 获取边界框的数量
    num_boxes = boxes.shape[0]

    # 计算每个边界框的面积
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # 获取所有类别
    unique_labels = np.unique(labels)

    keep_indices = []  # 保留的边界框索引

    for label in unique_labels:
        # 获取当前类别的检测框
        class_indices = np.where(labels == label)[0]
        class_boxes = boxes[class_indices]
        class_scores = scores[class_indices]
        class_areas = areas[class_indices]

        # 根据得分对边界框进行排序（从高到低）
        order = class_scores.argsort()[::-1]

        while order.size > 0:
            # 选择得分最高的边界框
            idx = order[0]
            keep_indices.append(class_indices[idx])

            # 计算剩余边界框与当前选择的边界框的交并比 (IoU)
            xx1 = np.maximum(class_boxes[idx, 0], class_boxes[order[1:], 0])
            yy1 = np.maximum(class_boxes[idx, 1], class_boxes[order[1:], 1])
            xx2 = np.minimum(class_boxes[idx, 2], class_boxes[order[1:], 2])
            yy2 = np.minimum(class_boxes[idx, 3], class_boxes[order[1:], 3])

            # 计算交集的宽度和高度
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)

            # 计算交集的面积
            inter = w * h

            # 计算交并比 (IoU)
            iou = inter / (class_areas[idx] + class_areas[order[1:]] - inter)

            # 移除 IoU 大于阈值的边界框
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]  # +1 是因为 order[0] 已经被处理过了

    return np.array(keep_indices)

# 后处理
def postprocess(boxes, scores, labels, input_shape, ratio, pad, num_classes=80, conf_threshold=0.4, iou_threshold=0.7):
    #pad , [0]width, [1]height
    #ratio,
    # 应用置信度阈值
    mask = scores > conf_threshold
    # boxes = boxes[1, mask, :]
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    # 调整边界框坐标至原图大小
    boxes[:, 0::2] -= pad[0]
    boxes[:, 1::2] -= pad[1]
    boxes[:, :] /= ratio[0]
    boxes[:, 0] = np.clip(boxes[:, 0], a_min=0, a_max=1280)
    boxes[:, 1] = np.clip(boxes[:, 1], a_min=0, a_max=720)
    boxes[:, 2] = np.clip(boxes[:, 2], a_min=0, a_max=1280)
    boxes[:, 3] = np.clip(boxes[:, 3], a_min=0, a_max=720)



    # 应用NMS
    # indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold)

    selected_indices = nms(boxes, scores, labels, iou_threshold)

    # 构建最终的检测结果列表
    detections = []
    for i in range(len(boxes)):
        box = boxes[i, :4]
        score = scores[i]
        class_id = labels[i]
        detections.append({
            'class_id': class_id,
            'score': float(score),
            'bbox': [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
        })

    return detections

def draw_boxes(image, detections):
    for detection in detections:
        bbox = detection['bbox']
        class_id = detection['class_id']
        score = detection['score']

        # 获取边界框坐标
        x1, y1, x2, y2 = map(int, bbox)

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制类别标签
        label = f'{class_id}: {score:.2f}'
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_rect = (x1, y1 - label_size[1] - 5, x1 + label_size[0], y1)
        cv2.rectangle(image, label_rect[:2], label_rect[2:], (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return image

# 主函数
def main():
    # 加载插件
    plugin = load_plugin(plugin_path, "grid_sampler")
    with open(engine_file, "rb") as f:
        serialized_engine = f.read()
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = create_context(engine)

    # 预处理图像
    image, ratio, pad, original_image = preprocess_image(image_file)    # inputs = {input_name: np.ascontiguousarray(image)}

    inputs, outputs, bindings, stream = common.allocate_buffers(engine)

    np.copyto(inputs[0].host, np.ascontiguousarray(image).ravel())

    # 运行推理
    # outputs = infer(engine, context, inputs, input_name, output_names)
    trt_outputs = common.do_inference(
            context=context,
            bindings=bindings,
            inputs=inputs,
            outputs=outputs,
            stream=stream
        )

    # 后处理
    boxes_output = outputs[0].host  # dets
    # boxes_output = np.reshape(boxes_output, [900, 4])
    boxes_scores = np.reshape(boxes_output, [300, 5])
    boxes = boxes_scores[:, :4]
    scores = boxes_scores[:, 4]
    # scores_output = outputs[1].host
    labels = outputs[1].host

    # 选择前300个最大值及其索引
    # k = 300
    # top_k_indices = scores_output.argsort()[-k:][::-1]  # 获取前300个最大值的索引，并按降序排列
    # scores = scores_output[top_k_indices]  # 获取前300个最大值
    # labels = top_k_indices % 80
    # indices = top_k_indices // 80
    # boxes = boxes_output[indices]
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    # 计算归一化的 x1, y1, x2, y2
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    # 将归一化的 x1, y1, x2, y2 转换为实际的图像坐标
    x1 = x1 * 640
    y1 = y1 * 640
    x2 = x2 * 640
    y2 = y2 * 640

    # 确保 x1, y1, x2, y2 的值在 [0, 640] 范围内
    x1 = np.clip(x1, a_min=0, a_max=640)
    y1 = np.clip(y1, a_min=0, a_max=640)
    x2 = np.clip(x2, a_min=0, a_max=640)
    y2 = np.clip(y2, a_min=0, a_max=640)

    # 将结果组合成一个新的矩阵
    xyxy = np.stack((x1, y1, x2, y2), axis=1)
    print()
    detections = postprocess(xyxy, scores, labels, image.shape[:2], ratio, pad)

    # 读取原始图像
    original_image = cv2.imread(image_file)

    result_image = draw_boxes(original_image, detections)

    # 显示结果图像
    cv2.imwrite('Result.jpg', result_image)

if __name__ == '__main__':
    main()

# class ModelData(object):
#     # MODEL_PATH = "ResNet50.onnx"
#     INPUT_SHAPE = (3, 640, 640)
#     # We can convert TensorRT data types to numpy types with trt.nptype()
#     DTYPE = trt.float32
#
# def load_normalized_test_case(test_image, pagelocked_buffer):
#     # Converts the input image to a CHW Numpy array
#     def normalize_image(image):
#         # Resize, antialias and transpose the image to CHW.
#         c, h, w = ModelData.INPUT_SHAPE
#         image_arr = (
#             np.asarray(image.resize((w, h), Image.LANCZOS))
#             .transpose([2, 0, 1])
#             .astype(trt.nptype(ModelData.DTYPE))
#             .ravel()
#         )
#         # This particular ResNet50 model requires some preprocessing, specifically, mean normalization.
#         return (image_arr / 255.0)
#
#     # Normalize the image and copy to pagelocked memory.
#     np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image)))
#     return test_image
#
#
# def non_max_suppression(boxes, scores, iou_threshold=0.45):
#     # Perform non-max suppression
#     indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.25, nms_threshold=iou_threshold)
#     return indices.flatten()
#
#
# def plot_bounding_boxes(image, boxes, labels, colors=None):
#     if colors is None:
#         colors = np.random.randint(0, 255, size=(80, 3), dtype='uint8')  # 80 classes
#
#     for i, (x, y, w, h) in enumerate(boxes):
#         color = colors[labels[i]].tolist()
#         cv2.rectangle(image, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), color, 2)
#         text = f'{labels[i]}'
#         cv2.putText(image, text, (int(x - w / 2), int(y - h / 2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#
#
# def process_output(output, image, conf_threshold=0.25, iou_threshold=0.7):
#     # Convert the output to a NumPy array
#     output = output.squeeze(0)  # Remove batch dimension
#
#     # Extract bounding boxes and class scores
#     bboxes = output[:, :4]
#     class_scores = output[:, 4:]
#
#     # Calculate the confidence score as the maximum class probability
#     scores = np.max(class_scores, axis=1)
#     class_preds = np.argmax(class_scores, axis=1)
#
#     # Filter out low-confidence predictions
#     high_conf_indices = scores > conf_threshold
#     bboxes = bboxes[high_conf_indices]
#     scores = scores[high_conf_indices]
#     class_preds = class_preds[high_conf_indices]
#
#     # Apply non-max suppression
#     keep_indices = non_max_suppression(bboxes, scores, iou_threshold)
#
#     # Filter boxes and labels
#     filtered_bboxes = bboxes[keep_indices]
#     filtered_labels = class_preds[keep_indices]
#
#     # Adjust bounding box coordinates to the original image size
#     original_height, original_width = image.shape[:2]
#     input_size = 640  # Input size to the model
#
#     scale_factor = max(original_width, original_height) / input_size
#
#     adjusted_bboxes = np.copy(filtered_bboxes)
#     adjusted_bboxes[:, 0] *= scale_factor  # x center
#     adjusted_bboxes[:, 1] *= scale_factor  # y center
#     adjusted_bboxes[:, 2] *= scale_factor  # width
#     adjusted_bboxes[:, 3] *= scale_factor  # height
#
#     # Ensure the bounding boxes fit within the image boundaries
#     adjusted_bboxes[:, 0] = np.clip(adjusted_bboxes[:, 0], 0, original_width)
#     adjusted_bboxes[:, 1] = np.clip(adjusted_bboxes[:, 1], 0, original_height)
#     adjusted_bboxes[:, 2] = np.clip(adjusted_bboxes[:, 2], 0, original_width)
#     adjusted_bboxes[:, 3] = np.clip(adjusted_bboxes[:, 3], 0, original_height)
#
#     # Plot bounding boxes
#     plot_bounding_boxes(image, filtered_bboxes, filtered_labels)
#
#     return image
#
#
# def main():
#     # engine = build_engine_onnx(model_file)
#     with open("yolov11x_halfF_simplifyF.engine", "rb") as f:
#         serialized_engine = f.read()
#     runtime = trt.Runtime(TRT_LOGGER)
#     engine = runtime.deserialize_cuda_engine(serialized_engine)
#     inputs, outputs, bindings, stream = common.allocate_buffers(engine)
#     context = engine.create_execution_context()
#     test_case = load_normalized_test_case(test_image, inputs[0].host)
#     trt_outputs = common.do_inference(
#         context,
#         engine=engine,
#         bindings=bindings,
#         inputs=inputs,
#         outputs=outputs,
#         stream=stream,
#     )
#     # 假设 output 是你的原始输出数组
#     output = trt_outputs[0]
#
#     # 转换维度
#     output = output.reshape(1, 84, 8400)
#
#     image = cv2.imread(test_image)  # Load your image
#     result_image = process_output(np.transpose(output, (0, 2, 1)), image)
#
#     # 保存图像
#     cv2.imwrite('detection_result.jpg', result_image)
#
# if __name__ == "__main__":
#     main()