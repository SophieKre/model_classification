from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from my_model_vgg import modelVGG, load_model, save_model

# Создаем приложение FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model_path = "mineral_vgg.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modelVGG = models.vgg16()

# Freeze parameters so we don't backprop through them
# for param in modelVGG.parameters():
#     param.requires_grad = False
class_names = ["biotite", "bornite", "chrysocolla", "malachite", "muscovite", "pyrite", "quartz"]
mineral_data = {
    "biotite": {
        "description": "Biotite is a common phyllosilicate mineral within the mica group.",
        "properties": ["Hardness: 2.5-3", "Color: Dark brown to black", "Luster: Vitreous"]
    },
    "bornite": {
        "description": "Bornite is an important copper ore mineral.",
        "properties": ["Hardness: 3", "Color: Brown to copper-red", "Luster: Metallic"]
    },
    "chrysocolla": {
        "description": "Chrysocolla is a hydrated copper phyllosilicate mineral.",
        "properties": ["Hardness: 2.5-3.5", "Color: Blue-green", "Luster: Vitreous to dull"]
    },
    "malachite": {
        "description": "Malachite is a copper carbonate hydroxide mineral.",
        "properties": ["Hardness: 3.5-4", "Color: Green", "Luster: Silky, adamantine, or dull"]
    },
    "muscovite": {
        "description": "Muscovite is a phyllosilicate mineral of aluminium and potassium.",
        "properties": ["Hardness: 2-2.5", "Color: Colorless to shades of green", "Luster: Pearly to vitreous"]
    },
    "pyrite": {
        "description": "Pyrite is a sulfide mineral also known as fool's gold.",
        "properties": ["Hardness: 6-6.5", "Color: Pale brass-yellow", "Luster: Metallic"]
    },
    "quartz": {
        "description": "Quartz is a hard, crystalline mineral composed of silicon and oxygen atoms.",
        "properties": ["Hardness: 7", "Color: Colorless through various colors", "Luster: Vitreous"]
    }
}

# Функция для перевода предсказания в конкретный класс
def get_class_from_prediction(prediction):
    # Используйте torch.max, чтобы получить индекс максимального значения вероятности
    _, predicted_index = torch.max(prediction, dim=1)
    # Получите название класса по индексу
    predicted_class = class_names[predicted_index.item()]
    return predicted_class
#vgg16
modelVGG.classifier = nn.Sequential(nn.Linear(in_features=25088, out_features=4096),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(in_features=4096, out_features=1000),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(in_features=1000, out_features=500),
                                 nn.Linear(500, 7),
                                 nn.LogSoftmax(dim=1))

modelVGG.load_state_dict(torch.load(model_path, map_location=device))
modelVGG.to(device)
modelVGG.eval()  # Переводим модель в режим инференса
# Функция предобработки изображений
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode == "RGBA":
        image = image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    content = await file.read()
    input_data = preprocess_image(content).to(device)

    # Отключаем вычисление градиентов
    with torch.no_grad():
        prediction = modelVGG(input_data)

    predicted_class = get_class_from_prediction(prediction)

    # Информация о минерале (заглушка)
    mineral_info = mineral_data.get(predicted_class, {
        "description": "No information available.",
        "properties": []
    })


    response_data = {
        "predicted_class": predicted_class,
        "mineral_info": mineral_info
    }

    return JSONResponse(content=response_data)

# Отображение главной страницы
@app.get("/")
def main():
    html_content = """
  <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Mineral Image Classifier</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                background: linear-gradient(135deg, #ffcc33, #ff6666);
                color: #333;
                text-align: center;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
                text-align: center;
                max-width: 500px;
                width: 100%;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            h1 {
                font-size: 24px;
                margin-bottom: 20px;
                color: #444;
            }
            form {
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            input[type="file"] {
                margin-bottom: 20px;
                padding: 10px;
                border: 2px solid #ffcc33;
                border-radius: 5px;
                background-color: #fff;
                cursor: pointer;
            }
            input[type="submit"] {
                padding: 10px 20px;
                font-size: 16px;
                background-color: #ffcc33;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }
            input[type="submit"]:hover {
                background-color: #ff9900;
            }
            .prediction {
                margin-top: 20px;
                font-size: 18px;
                color: #555;
            }
            .mineral-info {
                margin-top: 10px;
                font-size: 16px;
                color: #777;
            }
            .property-list {
                list-style: none;
                padding: 0;
                margin: 10px 0;
            }
            .property-list li {
                background-color: #ffcc33;
                padding: 5px 10px;
                border-radius: 5px;
                margin: 5px 0;
                color: #333;
            }
        </style>
        <script>
    async function uploadImage(event) {
        event.preventDefault();
        console.log("Uploading image...");
        const fileInput = document.querySelector('input[type="file"]');
        const formData = new FormData();
        formData.append("file", fileInput.files[0]);

        try {
            const response = await fetch("/upload", {
                method: "POST",
                body: formData
            });
            const data = await response.json();
            console.log(data); // Log the response data
            document.querySelector('.prediction').textContent = 'Предсказанный класс минерала: ' + data.predicted_class;
            document.querySelector('.mineral-info').textContent = 'Описание: ' + data.mineral_info.description;
        } catch (error) {
            console.error("Error uploading image:", error);
        }
    }
</script>

    </head>
    <body>
        <div class="container">
            <h1>Загрузите изображение геологического образца для предсказания</h1>
            <form onsubmit="uploadImage(event)">
                <input type="file" name="file" accept="image/*" required>
                <input type="submit" value="Загрузить">
            </form>
            <div class="prediction"></div>
            <div class="mineral-info"></div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Запуск FastAPI-приложения
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
