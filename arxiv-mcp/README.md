<h1 align="center">ArXiv-MCP: Academic Paper Search for AI Agents</h1>

<p align="center">
  <img src="https://info.arxiv.org/brand/images/brand-logo-primary.jpg" alt="arXiv Logo" width="400">
</p>

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server implementation that enables AI agents to search, retrieve, and analyze academic papers from [arXiv](https://arxiv.org), the popular open-access repository of electronic preprints.

## Overview

This project provides an MCP server that allows AI agents to interact with the arXiv repository, search for papers based on keywords, retrieve detailed information about specific papers, and even extract and analyze the content of papers. It serves as both a practical tool for research assistance and a reference implementation for building MCP servers.

The implementation follows the best practices laid out for building MCP servers, allowing seamless integration with any MCP-compatible client.

## Features

The server provides several powerful tools for academic research:

1. **`search_papers`**: Search for papers on arXiv using keywords and get comprehensive summaries
2. **`get_paper_details`**: Retrieve detailed information about a specific paper by its arXiv ID
3. **`extract_paper_content`**: Download and extract the full text content from a paper's PDF
4. **`analyze_paper`**: Analyze a paper's content and generate a comprehensive summary

## Prerequisites

- Python 3.11+
- Docker if running the MCP server as a container (recommended)

## Installation

### Using uv

1. Install uv if you don't have it:
   ```bash
   pip install uv
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/kelvingao/arxiv-mcp.git
   cd arxiv-mcp
   ```

3. Install dependencies:
   ```bash
   uv pip install -e .
   ```

4. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```

5. Configure your environment variables in the `.env` file (see Configuration section)

### Using Docker (Recommended)

1. Build the Docker image:
   ```bash
   docker build -t mcp/arxiv --build-arg PORT=8050 .
   ```

2. Create a `.env` file based on `.env.example` and configure your environment variables

## Configuration

The following environment variables can be configured in your `.env` file:

| Variable | Description | Example |
|----------|-------------|---------|
| `TRANSPORT` | Transport protocol (sse or stdio) | `sse` |
| `HOST` | Host to bind to when using SSE transport | `0.0.0.0` |
| `PORT` | Port to listen on when using SSE transport | `8050` |

## Running the Server

### Using uv

#### SSE Transport

```bash
# Set TRANSPORT=sse in .env then:
python src/server.py
```

The MCP server will run as an API endpoint that you can connect to with the configuration shown below.

#### Stdio Transport

With stdio, the MCP client itself can spin up the MCP server, so nothing to run at this point.

### Using Docker

#### SSE Transport

```bash
docker run --env-file .env -p 8050:8050 mcp/arxiv
```

The MCP server will run as an API endpoint within the container that you can connect to with the configuration shown below.

#### Stdio Transport

With stdio, the MCP client itself can spin up the MCP server container, so nothing to run at this point.

## Integration with MCP Clients

### SSE Configuration

Once you have the server running with SSE transport, you can connect to it using this configuration:

```json
{
  "mcpServers": {
    "arxiv": {
      "transport": "sse",
      "url": "http://localhost:8050/sse"
    }
  }
}
```

> **Note for Windsurf users**: Use `serverUrl` instead of `url` in your configuration:
> ```json
> {
>   "mcpServers": {
>     "arxiv": {
>       "transport": "sse",
>       "serverUrl": "http://localhost:8050/sse"
>     }
>   }
> }
> ```

> **Note for n8n users**: Use host.docker.internal instead of localhost since n8n has to reach outside of its own container to the host machine:
> 
> So the full URL in the MCP node would be: http://host.docker.internal:8050/sse

Make sure to update the port if you are using a value other than the default 8050.

### Python with Stdio Configuration

Add this server to your MCP configuration for Claude Desktop, Windsurf, or any other MCP client:

```json
{
  "mcpServers": {
    "arxiv": {
      "command": "your/path/to/arxiv-mcp/.venv/bin/python",
      "args": ["your/path/to/arxiv-mcp/src/main.py"],
      "env": {
        "TRANSPORT": "stdio"
      }
    }
  }
}
```

### Docker with Stdio Configuration

```json
{
  "mcpServers": {
    "arxiv": {
      "command": "docker",
      "args": ["run", "--rm", "-i", 
               "-e", "TRANSPORT", 
               "mcp/arxiv"],
      "env": {
        "TRANSPORT": "stdio"
      }
    }
  }
}
```

## Usage Examples

Here are some examples of how to use the arXiv MCP server with an AI agent:

### Searching for Papers

```
Find recent papers about quantum computing published in the last year.
```

### Getting Paper Details

```
Get details for the paper with arXiv ID 2303.08774
```

### Extracting Paper Content

```
Extract the full text from the paper with arXiv ID 2303.08774
```

### Analyzing a Paper

```
Analyze the methodology section of the paper with arXiv ID 2303.08774
```

## Building Your Own MCP Server

This implementation provides a foundation for building more complex MCP servers. To build your own:

1. Add your own tools by creating methods with the `@mcp.tool()` decorator
2. Create your own lifespan function to add your own dependencies (clients, database connections, etc.)
3. Modify the existing tools or add new ones to enhance functionality
4. Add prompts and resources with `@mcp.resource()` and `@mcp.prompt()`

## License

[MIT License](LICENSE)

## Acknowledgements

- [arXiv](https://arxiv.org) for providing open access to research papers
- The [Model Context Protocol](https://modelcontextprotocol.io) team for creating the MCP standard