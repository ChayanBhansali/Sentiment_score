# Text Analysis Dashboard
![Demo Video](https://github.com/ChayanBhansali/Sentiment_score/blob/main/video.gif)
## Installation and Setup

### Option 1: Running with Docker Compose (Recommended)


#### Steps

1. **Install Docker and Docker Compose**: Ensure you have Docker and Docker Compose installed on your machine. You can download them from [Docker's official website](https://www.docker.com/get-started).


2. **Clone the Repository**
    ```sh
    git clone https://github.com/ChayanBhansali/Sentiment_score.git
    cd Sentiment_score
    ```

3. **Environment Setup** (optional)
   ```sh
   # Conda
   conda create -n sentiment-env python=3.9
   conda activate sentiment-env

   # Or Python venv
   python3 -m venv sentiment-env
   source sentiment-env/bin/activate  # Unix
   sentiment-env\Scripts\activate     # Windows
   ```

4. **Build and Run Containers**
    ```sh
    docker-compose up --build
    ```

5. **Access the Application**
    Open your web browser and navigate to `http://localhost:7860`

6. **Stop the Containers**
    ```sh
    docker-compose down
    ```

### Option 2: Running with Uvicorn 


#### Steps
1. **Clone the Repository**
    ```sh
    git clone https://github.com/ChayanBhansali/Sentiment_score.git
    cd Sentiment_score
    ```

2. **Create a Virtual Environment** (Recommended)
    

3. **Install Dependencies**
    ```sh
    pip install -r requirements.txt
    ```

4. **Run the Application**
    ```sh
    uvicorn src.web:app --host 0.0.0.0 --port 7860
    ```

5. **Access the Application**
    Open your web browser and navigate to `http://localhost:7860`

## Visualization
The spider graph comprises five metrics: four derived from emotion analyses and one representing an education score. Each point is scaled from 0 to 1, allowing for a comprehensive comparative visualization of the data points.


