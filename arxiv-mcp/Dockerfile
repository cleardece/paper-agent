FROM python:3.12-slim

ARG PORT=8050

WORKDIR /app

# Install uv
RUN pip install uv

# Copy the MCP server files
COPY . .

RUN python -m venv .venv
RUN .venv/bin/pip install -e .

EXPOSE ${PORT}

# Set PORT environment variable
ENV PORT=${PORT}
ENV TRANSPORT=sse

# Command to run the MCP server
CMD [".venv/bin/python", "src/server.py"]
