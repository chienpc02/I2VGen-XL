import os
os.system('pip install -r requirements.txt')
# os.system('pip install "modelscope" --upgrade -f https://pypi.org/project/modelscope/')
os.system('pip install xformers==0.0.20')

max_try=5
while max_try>0:
    state = os.system("git clone https://github.com/modelscope/modelscope.git")
    if state == 0:
        break
    max_try -= 1
os.chdir('modelscope')
os.system('pip install .')

import gradio as gr
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys

image_to_video_pipe = pipeline(task="image-to-video", model='damo/Image-to-Video', model_revision='v1.1.0', device='cuda:0')
video_to_video_pipe = pipeline(task="video-to-video", model='damo/Video-to-Video', model_revision='v1.1.0', device='cuda:0')

def upload_file(file):
    return file.name

def image_to_video(image_in):
    if image_in is None:
        raise gr.Error('Vui lòng tải hình ảnh lên hoặc đợi quá trình tải hình ảnh hoàn tất')
    print(image_in)
    output_video_path = image_to_video_pipe(image_in, output_video='./i2v_output.mp4')[OutputKeys.OUTPUT_VIDEO]
    print(output_video_path)
    return output_video_path


def video_to_video(video_in, text_in):
    if video_in is None:
        raise gr.Error('Vui lòng hoàn thành bước đầu tiên')
    if text_in is None:
        raise gr.Error('Vui lòng nhập mô tả văn bản')
    p_input = {
            'video_path': video_in,
            'text': text_in
        }
    output_video_path = video_to_video_pipe(p_input, output_video='./v2v_output.mp4')[OutputKeys.OUTPUT_VIDEO]
    print(output_video_path)
    return output_video_path


with gr.Blocks() as demo:
    gr.Markdown(
        """<center><font size=7>I2VGen-XL Demo</center>
        <left><font size=3>I2VGen-XLcó thể tạo các video có mục tiêu tương tự và cùng ngữ nghĩa dựa trên hình ảnh tĩnh và văn bản do người dùng nhập vào. Các video được tạo ra có độ phân giải cao(1280 * 720)、Các tính năng bao gồm màn hình rộng (16:9), thời gian mạch lạc và chất lượng tốt.</left>

        <left><font size=3>I2VGen-XL can generate videos with similar contents and semantics based on user input static images and text. The generated videos have characteristics such as high-definition (1280 * 720), widescreen (16:9), coherent timing, and good texture.</left>
        """
    )
    with gr.Box():
        gr.Markdown(
        """<left><font size=3>Step 1: Chọn hình ảnh phù hợp để tải lên (tỷ lệ hình ảnh khuyến nghị là 1:1), sau đó nhấp vào "Tạo video" và chuyển sang bước tiếp theo sau khi có được video ưng ý.</left>

        <left><font size=3>Step 1: Select the image to upload (it is recommended that the image ratio is 1:1), and then click on “Generate Video” to obtain a generated video before proceeding to the next step.</left>"""
        )
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(label="Input Image", type="filepath", interactive=False, elem_id="image-in", height=300)
                with gr.Row():
                    upload_image = gr.UploadButton("Upload Image", file_types=["image"], file_count="single")
                    image_submit = gr.Button("Create Video🎬")
            with gr.Column():
                video_out_1 = gr.Video(label='video được tạo', elem_id='video-out_1', interactive=False, height=300)
    with gr.Box():
        gr.Markdown(
        """<left><font size=3>Step 2: Bổ sung mô tả văn bản tiếng Anh cho nội dung video, sau đó nhấp vào "Tạo video có độ phân giải cao". Quá trình tạo video mất khoảng 2 phút.</left>

        <left><font size=3>Step 2: Add the English text description of the video you want to generate, and then click on "Generate high-resolution video". The video generation will take about 2 minutes.</left>"""
        )
        with gr.Row():
            with gr.Column():
                text_in = gr.Textbox(label="mô tả văn bản", lines=2, elem_id="text-in")
                video_submit = gr.Button("Tạo video có độ phân giải cao🎥")
            with gr.Column():
                video_out_2 = gr.Video(label='video được tạo', elem_id='video-out_2', height=300)
    gr.Markdown("<left><font size=2>注：Nếu video đã tạo không thể phát được, vui lòng thử nâng cấp trình duyệt của bạn hoặc sử dụng trình duyệt Chrome.</left>")


    upload_image.upload(upload_file, upload_image, image_in, queue=False)
    image_submit.click(fn=image_to_video, inputs=[image_in], outputs=[video_out_1])
    video_submit.click(fn=video_to_video, inputs=[video_out_1, text_in], outputs=[video_out_2])

demo.queue(status_update_rate=1, api_open=False).launch(share=False, show_error=True)