### Using this frontend

> From inside this `frontend` folder:

You can serve the frontend with `streamlit run app.py` (default port is `8501`). Alternatively, you case use Makefile with `make run_ui`.

❗ Note that you should have the required packages with `make install_requirements`
❗❗ Also recall that you need to create your .env. Use .env.sample to start.

## Using this frontend with Docker

The `frontend` has corresponding `Dockerfile`s for the web UI.

1. To create a Docker image, inside the corresponding folder run `docker built -t NAME_FOR_THE_UI_IMAGE .`
2. Run a container for `frontend` with `docker run -p MACHINE_PORT:CONTAINER_PORT NAME_FOR_THE_UI_IMAGE`;

  Here, `MACHINE_PORT` is the `localhost` port you want to link to the container, while `CONTAINER_PORT` is the port which will be used by the running app in the container.


3. ❗ You won't be able to reach the API container through `localhost`; You'll need to [link](https://docs.docker.com/network/links/) the containers:

  * **UI:** `docker run -p 8501:8501 --link api:api NAME_FOR_THE_UI_IMAGE`

  ❗ See the corresponding instructions when it comes to dockerize the API
  ❗❗ Note that Docker docs mention that `--link` might be removed in the future (as of 2022.06). Alternatives can be [user-defined bridges](https://docs.docker.com/network/bridge/#differences-between-user-defined-bridges-and-the-default-bridge) or [Docker Compose](https://docs.docker.com/compose/)
