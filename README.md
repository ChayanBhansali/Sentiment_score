# Sentiment Score

To run the application using Docker Compose, follow these steps:

1. **Install Docker and Docker Compose**: Ensure you have Docker and Docker Compose installed on your machine. You can download them from [Docker's official website](https://www.docker.com/get-started).

2. **Clone the Repository**: Clone the repository

    ```sh
    git clone https://github.com/ChayanBhansali/Sentiment_score.git
    cd Sentiment_score
    ```

3. **Configure environment variables**: Create a `.env` file in the root directory of the project and configure `DATABASE_URL` (optional).

4. **Build and Run the Containers**: Use Docker Compose to build and run the containers.

    ```sh
    docker-compose up --build
    ```

5. **Access the Application**: Once the containers are up and running, you can access the application at `http://localhost:8000`.

6. **Stopping the Containers**: To stop the containers:

    ```sh
    docker-compose down
    ```
# Demo Video
![Demo Video](https://github.com/ChayanBhansali/Sentiment_score/blob/main/video.gif)
