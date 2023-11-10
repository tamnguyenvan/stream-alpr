# Stream ALPR
This project is developed based on Savant. Savant is a framework/wrapper that facilitates the deployment of DeepStream applications in the most convenient and straightforward manner possible. Stream ALPR aims to enable users to quickly deploy an ALPR (Automatic License Plate Recognition) system with just a few simple steps. This system is capable of concurrently processing multiple streams at once, making it easily applicable in a production environment.

## Requirements
The only requirement is that your machine must have a GPU and Nvidia driver installed. Check the supported GPU types here. Additionally, you also need to install Docker and Nvidia Docker runtime.

## Quick Start
After satisfying the above requirements, you can easily launch the module with the following command:
```bash
docker compose -f docker-compose.yml up
```

## TODOs
- [ ] Allow adding custom models
- [ ] Add documentation related to the models

## Contributions
All contributions to the project are welcome.

## Acknowledgments
A sincere thank you to the Savant team for developing such an excellent framework.