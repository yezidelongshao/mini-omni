import flask
import base64
import tempfile
import traceback
from flask import Flask, Response, stream_with_context
from inference import OmniInference


class OmniChatServer(object):
    def __init__(self, ip='0.0.0.0', port=60808, run_app=True,
                 ckpt_dir='./checkpoint', device='cuda:0') -> None:
        server = Flask(__name__)
        # CORS(server, resources=r"/*")
        # server.config["JSON_AS_ASCII"] = False

        self.client = OmniInference(ckpt_dir, device)
        self.client.warm_up()
        #定义路由，设置chat路径来处理POST请求，并将请求交由chat()方法处理
        server.route("/chat", methods=["POST"])(self.chat)
        
        if run_app:
            #运行Flask应用
            server.run(host=ip, port=port, threaded=False)
        else:
            #只初始化Flask应用而不运行
            self.server = server

    def chat(self) -> Response:
        #获取请求数据
        req_data = flask.request.get_json()
        try:
            #处理音频数据，获取音频数据并将其转换为字节数据
            data_buf = req_data["audio"].encode("utf-8")
            #解码数据为二进制数据
            data_buf = base64.b64decode(data_buf)
            stream_stride = req_data.get("stream_stride", 4)
            max_tokens = req_data.get("max_tokens", 2048)
            #将音频文件写入临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(data_buf)
                #调用OmniInference类实例的run_AT_batch_stream方法生成音频流返回给客户端
                audio_generator = self.client.run_AT_batch_stream(f.name, stream_stride, max_tokens)
                return Response(stream_with_context(audio_generator), mimetype="audio/wav")
        except Exception as e:
            print(traceback.format_exc())


# CUDA_VISIBLE_DEVICES=1 gunicorn -w 2 -b 0.0.0.0:60808 'server:create_app()'
def create_app():
    # 创建OmniChatServer实例
    server = OmniChatServer(run_app=False)
    return server.server


def serve(ip='0.0.0.0', port=60808):
    #直接调用服务，并将监听ip和端口绑定
    OmniChatServer(ip, port=port, run_app=True)


if __name__ == "__main__":
    #这种结构常用于让 Python 文件既可以作为独立的脚本运行，又可以被其他模块导入并复用，而不会在导入时执行特定的主程序代码
    import fire
    fire.Fire(serve)
    
