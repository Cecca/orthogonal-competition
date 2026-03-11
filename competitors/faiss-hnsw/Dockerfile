# -----------------------------------------------------------------------
# NNS Competition – Student Dockerfile
#
# Build and push:
#   docker build -t <your-team>/nns:latest .
# -----------------------------------------------------------------------

FROM ann-orthogonal/base:latest

# ---- Install your algorithm's dependencies ----
RUN pip install --no-cache-dir faiss-cpu

# ---- Copy your algorithm and scenario definitions ----
COPY algorithm.py  /app/algorithm.py
COPY scenarios.yaml /app/scenarios.yaml

# No ENTRYPOINT override – the base image harness handles everything.
