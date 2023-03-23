import tensorrt as trt
import torch
import time
import imageio
from options import args
import utility
import os
import tqdm
import shutil
import streamlit



class Tensorrt:
    def __init__(self, args):
        self.args = args
        self.scale = args.scale
        self.lr_path = args.lr_path
        self.hr_path = args.hr_path
        self.onnxfile = args.onnxfile
        self.enginefile = args.enginefile
        print("初始化trt引擎...")
        self.trt = trt
        # 创建记录器
        self.logger = trt.Logger(trt.Logger.WARNING)
        # 记录日志
        self.log = {'time': [], 'path': [], 'images': [], 'psnr': [], 'ssim': []}
        # 路径检查和创建
        self.log_path = os.path.join(self.args.results, self.args.model)
        self.img_path = os.path.join(self.args.results, self.args.model, 'x' + str(self.scale), self.args.precision)

        if os.path.exists(self.img_path):
            if self.args.save:
                shutil.rmtree(self.img_path)
                os.makedirs(self.img_path)
        else:
            os.makedirs(self.img_path)


    def onnxtotrt(self):
        # 创建构建器
        builder = self.trt.Builder(self.logger)
        # 预创建网络
        network = builder.create_network(1 << int(self.trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        # 加载onnx解析器和测试
        parser = self.trt.OnnxParser(network, self.logger)
        success = parser.parse_from_file(self.args.onnxfile)
        for idx in range(parser.num_errors):
            print(parser.get_error(idx))
        if not success:
            print("onnx文件打开失败或不存在！")
            exit(1)

        # 创建配置文件
        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        profile.set_shape("input", (1, 3, 48, 48), (1, 3, 256, 256), (1, 3, 1024, 1024))
        config.add_optimization_profile(profile)

        # 设置FP16精度
        if self.args.precision == 'FP16':
            config.set_flag(self.trt.BuilderFlag.FP16)

        # 序列化引擎并保存到文件
        serialized_engine = builder.build_serialized_network(network, config)
        print(self.enginefile)
        with open(self.args.enginefile, 'wb') as f:
            f.write(serialized_engine)

    # 加载引擎并反序列化
    def load_engine(self):
        self.enginefile = os.path.join('.\models\engines',
                                       self.args.model + '_' + self.args.precision
                                       + 'x' + str(self.args.scale) + '.engine')
        print(self.enginefile)
        if not os.path.exists(self.enginefile):
            self.onnxtotrt()
        runtime = self.trt.Runtime(self.logger)
        with open(self.enginefile, 'rb') as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        context = engine.create_execution_context()
        context.active_optimization_profile = 0
        return context

    # 读取图像
    def read_images(self):
        lr_images = []
        hr_images = []
        lr_list = sorted(os.listdir(self.args.lr_path))
        # 读取hr图像
        if os.path.exists(self.args.hr_path):
            hr_list = sorted(os.listdir(self.args.hr_path))
            if len(hr_list) == len(lr_list):
                for i in hr_list:
                    hr_path = os.path.join(args.hr_path, i)
                    image_hr = imageio.v2.imread(hr_path, pilmode='RGB')
                    image_hr_to_tensor = torch.Tensor(image_hr).permute(2, 0, 1).unsqueeze(0).cuda()
                    image_hr_to_tensor = utility.crop_border(image_hr_to_tensor, self.scale)
                    hr_images.append(image_hr_to_tensor)
        # 读取lr图像
        for i in lr_list:
            # 记录图像名称
            self.log['images'].append(i)
            lr_path = os.path.join(args.lr_path, i)
            image_lr = imageio.v2.imread(lr_path, pilmode='RGB')
            image_lr_to_tensor = torch.Tensor(image_lr).permute(2, 0, 1).unsqueeze(0).cuda()
            lr_images.append(image_lr_to_tensor)

        return lr_images, hr_images

    # 执行推理
    def inference(self, context, input_list):
        print('推理中...')
        # 保存sr图像
        output_images = []
        for img_input in input_list:

            if self.args.model == 'IMDN' or self.args.model == 'BSNR':
                img_input = img_input / 255.0

            # 获取图像维度
            width = img_input.shape[2]
            height = img_input.shape[3]

            # 建立输入输出缓冲
            img_output = torch.empty(size=(1, 3, width * self.scale, height * self.scale), device='cuda')
            ptr_output = img_output.contiguous().data_ptr()
            prt_input = img_input.contiguous().data_ptr()
            bindings = [prt_input, ptr_output]

            # 设置输入维度
            context.set_binding_shape(0, (1, 3, width, height))

            # 执行推理
            t = time.time()
            context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)
            single_time = time.time() - t

            # 将输出图像添加到sr_images(需要先将输出图像有cuda转存到内存，否则会导致推理时间不稳定？)
            if self.args.model == 'IMDN' or self.args.model == 'BSNR':
                img_output = img_output * 255.0

            sr_image = img_output.cpu()
            output_images.append(sr_image)

            # 记录推理时间
            self.log['time'].append(single_time)

        # print('单张图像推理时间: ', self.log['time'])
        print('总推理时间: ', sum(self.log['time']))

        return output_images

    def images_processing(self, sr_images, hr_images):
        self.sr_images_list = []
        print('输出图像处理中...')
        for i in tqdm.tqdm(range(len(sr_images)), ncols=100):
            # sr图像量化
            sr_image = utility.quantize(sr_images[i], 255).cuda()
            # 如果hr图像存在，计算PSNR和SSIM
            if len(hr_images):
                # hr图像量化
                hr_image = utility.quantize(hr_images[i], 255)
                # 计算PSNR和SSIM
                self.log['psnr'].append(utility.calc_psnr(sr_image, hr_image, args.scale, 255))
                self.log['ssim'].append(utility.calc_ssim(sr_image, hr_image, args.scale))
            else:
                self.log['psnr'].append(0)
                self.log['ssim'].append(0)
            # sr图像由tensor转numpy
            sr_image = sr_image[0].data.byte().permute(1, 2, 0).cpu().numpy()
            self.sr_images_list.append(sr_image)
            # print(self.sr_images_list[i].shape)
            # 保存sr图像
            if self.args.save:
                # sr_image = sr_image[0].data.byte().permute(1, 2, 0).cpu().numpy()
                sr_name = 'x' + str(self.args.scale) + '_' + self.log['images'][i]
                imageio.imsave(os.path.join(self.img_path, sr_name), sr_image)

            # print(self.log['images'][i])
        print('执行完毕！')

        if os.path.exists(self.args.hr_path):
            # print('PSNR列表: ', trt.log['psnr'])
            print('PSNR平均值: ', sum(self.log['psnr']) / len(self.log['psnr']))
            # print('SSIM列表: ', trt.log['ssim'])
            print('SSIM平均值: ', sum(self.log['ssim']) / len(self.log['ssim']))
        print('推理时间: ', sum(self.log['time']))

    def write_log(self, t):
        total_time = time.time() - t
        log = os.path.join(self.log_path, (time.strftime('%Y.%m.%d_%H.%M.%S', time.localtime()) + '.log'))

        with open(log, 'w') as f:
            for arg in vars(self.args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            for i in range(len(self.log['images'])):
                f.write('\n{}\n[psnr: {:.3f}] [ssim: {:.4f}] [infer_time: {:.6f}]\n'.format(
                    self.log['images'][i], self.log['psnr'][i], self.log['ssim'][i], self.log['time'][i]))
            num_eval = len(self.log['psnr'])
            if num_eval:
                avg_psnr = sum(self.log['psnr']) / num_eval
                avg_ssim = sum(self.log['ssim']) / num_eval
            f.write('\navg_psnr: {:.4f}\navg_ssim: {:.5f}\n'.format(avg_psnr, avg_ssim))
            f.write('total_infer_time: {:.6f}\ntotal_time: {:.6f}'.format(sum(self.log['time']), total_time))

    def run(self):
        # 记录总时间
        t0 = time.time()
        # # 创建trt
        # trt = Tensorrt(args)
        # 加载引擎
        engine_context = self.load_engine()
        # 加载图像数据
        lr, hr = self.read_images()
        # 执行推理
        sr = self.inference(engine_context, lr)
        print('推理完成！')
        # 图像后处理
        self.images_processing(sr, hr)
        # 写入日志
        self.write_log(t0)
        print('执行时间：', time.time() - t0)
        return self.sr_images_list, self.log


if __name__ == '__main__':
    trt = Tensorrt(args)
    trt.run()


