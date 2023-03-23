import streamlit as st
from main import Tensorrt
from options import args
import re
import os


def startTRT():
    args.lr_path = os.path.join('.\images', datatest_selectbox, 'LR_bicubic', scale_selectbox)
    args.hr_path = os.path.join('.\images', datatest_selectbox, 'HR')
    args.precision = precision_selectbox
    scale_strtoint = int(re.sub('x', '', scale_selectbox))
    args.scale = scale_strtoint
    args.model = model_selectbox
    args.save = False

    trt = Tensorrt(args)
    sr_images, trt_log = trt.run()
    size = len(sr_images)

    l = []
    for i in range(size):
        l.append('{}--PSNR: {:.3f}--SSIM: {:.4f}'.format(trt_log['images'][i], trt_log['psnr'][i], trt_log['ssim'][i]))
    psnr_avg = sum(trt_log['psnr']) / size
    ssim_avg = sum(trt_log['ssim']) / size
    st.sidebar.write('PSNR平均值: {:.3f}'.format(psnr_avg))
    st.sidebar.write('SSIM平均值: {:.4f}'.format(ssim_avg))

    st.image(sr_images, caption=l)


# sr_images, trt_log = startTRT()


a = st.sidebar.button('开始',
                      key=None, help=None,
                      on_click=startTRT, args=None,
                      kwargs=None)

datatest_selectbox = st.sidebar.selectbox(
    label="请选择测试数据：",
    options=("Set5", "Set14", "B100", 'Urban100'),
    key="t0"
)

# if datatest_selectbox == "自定义数据":
#     file = st.sidebar.file_uploader('浏览文件')
#     st.sidebar.text(file)


model_selectbox = st.sidebar.selectbox(
    label="请选择模型：",
    options=("IMDN", "RLFN", "BSNR", 'EFDN', 'EFDN-F'),
    key="t1"
)

if model_selectbox == 'RLFN':
    scale_selectbox = st.sidebar.selectbox(
        label="请选择比例：",
        options=("x2", "x4"),
        key="t2"
    )
elif model_selectbox == 'EFDN-F':
    scale_selectbox = st.sidebar.selectbox(
        label="请选择比例：",
        options=("x4",),
        key="t2"
    )
else:
    scale_selectbox = st.sidebar.selectbox(
        label="请选择比例：",
        options=("x2", "x3", "x4"),
        key="t2")

precision_selectbox = st.sidebar.selectbox(
    label="请选择精度：",
    options=("FP16", "FP32"),
    key="t3"
)

#
# col1, col2, col3 = st.columns(3)
#
# with col1:
#     st.header("A cat")
#     st.image("https://static.streamlit.io/examples/cat.jpg")
#
# with col2:
#     st.header("A dog")
#     st.image("https://static.streamlit.io/examples/dog.jpg")
#
# with col3:
#     st.header("An owl")
#     st.image("https://static.streamlit.io/examples/owl.jpg")
#
# # st.text(add_selectbox2)

if st.sidebar.button("Clear All"):
    st.cache_resource.clear()
