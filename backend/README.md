### Using this backend

> From inside the `backend` folder:

You can serve the API with `uvicorn fast_api.api:app --reload` (default port is `8000`)

❗ Note that you should have the required packages with `make install_requirements`
❗❗ Also recall that you need to create your .env. Use .env.sample to start.

### Using this frontend with Docker

The `backend` has corresponding `Dockerfile`s for the API.

1. To create a Docker image, inside the corresponding folder run `docker built -t NAME_FOR_THE_API_IMAGE .`
2. Run a container for API with `docker run -p MACHINE_PORT:CONTAINER_PORT NAME_FOR_THE_API_IMAGE`;

  Here, `MACHINE_PORT` is the `localhost` port you want to link to the container, while `CONTAINER_PORT` is the port which will be used by the running app in the container.


3. ❗ You won't be able to reach the API container through `localhost`; You'll need to [link](https://docs.docker.com/network/links/) the containers:

  * **API:** `docker run -p 8000:8000 NAME_FOR_THE_API_IMAGE --name api`

  This way you can use `api` instead of `localhost` to reach the API container from the frontend

  ❗ See the corresponding instructions when it comes to dockerize the UI
  ❗❗ Note that Docker docs mention that `--link` might be removed in the future (as of 2022.06). Alternatives can be [user-defined bridges](https://docs.docker.com/network/bridge/#differences-between-user-defined-bridges-and-the-default-bridge) or [Docker Compose](https://docs.docker.com/compose/)
