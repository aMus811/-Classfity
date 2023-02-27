from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from werkzeug.utils import secure_filename
from datetime import timedelta
import os
from model import trash_classify

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])  # 设置允许的文件格式


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.send_file_max_age_default = timedelta(seconds=1)  # 设置静态文件缓存过期时间


@app.route('/', methods=['GET'])
def index():
    return render_template("Photo_selected.html")


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        upload_path = r'./static/images'  # 上传文件目录
        if not os.path.exists(upload_path):  # 判断文件夹是否存在
            os.makedirs(upload_path)

        filedata = request.files.get('fileField')  # 获取前端对象
        if not (filedata and allowed_file(filedata.filename)):  # 检查文件类型
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})
        file_path = os.path.join(upload_path, secure_filename(filedata.filename))  # 指定保存文件夹的路径
        print(file_path)
        file_name = "img.jpg"

        try:
            # 上传文件
            filedata.save(file_path)
            print(file_path, file_name, upload_path)
            listtest, a = trash_classify(file_path, file_name, upload_path)
            return render_template('Photo_selected_ok.html', possible=listtest, fpf=upload_path)
        except IOError:
            return jsonify({'code': -1, 'msg': '上传失败，请重试！'})
    else:
        return render_template('Photo_selected.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8987, debug=True, use_reloader=False)
