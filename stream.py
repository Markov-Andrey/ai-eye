from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import os

app = FastAPI()
os.makedirs('frames', exist_ok=True)

html = """
<!DOCTYPE html>
<html>
  <body>
    <h1>Захват видео и отправка на сервер</h1>

    <input type="file" id="videoInput" accept="video/*" capture="camera"><br><br>
    <video id="video" autoplay playsinline style="width: 400px; height: auto;"></video><br><br>
    <button id="send">Отправить на сервер</button>

    <script>
      const videoInput = document.getElementById('videoInput');
      const video = document.getElementById('video');
      const sendBtn = document.getElementById('send');
      let videoData = null;

      videoInput.addEventListener('change', () => {
        const file = videoInput.files[0];
        if (file) {
          const url = URL.createObjectURL(file);
          video.src = url;
          video.play();
        }
      });

      sendBtn.addEventListener('click', () => {
        if (videoInput.files.length === 0) {
          alert('Выберите видео файл');
          return;
        }

        const formData = new FormData();
        formData.append("file", videoInput.files[0]);

        fetch('http://192.168.100.6:8000/frame', {
          method: 'POST',
          body: formData
        }).then(() => {
          alert('Видео отправлено на сервер');
        });
      });
    </script>
  </body>
</html>
"""


@app.get("/")
async def get():
    return HTMLResponse(html)


@app.post("/frame")
async def receive_frame(request: Request):
    form = await request.form()
    video_file = form['file']
    file_path = os.path.join('frames', video_file.filename)

    with open(file_path, 'wb') as f:
        f.write(await video_file.read())

    print(f"Видео сохранено: {file_path}")
    return {"status": "ok"}
