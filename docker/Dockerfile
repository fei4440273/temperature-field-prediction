FROM ra-msml-pinn:local-cu124

WORKDIR /workspace
ENV PYTHONPATH=/workspace/src \
    MPLBACKEND=Agg \
    PATH=/home/pinn/.local/bin:$PATH

COPY --chown=pinn:pinn requirements.txt pyproject.toml README.md /workspace/
COPY --chown=pinn:pinn src /workspace/src
RUN /opt/conda/envs/PINN/bin/pip install \
    --no-cache-dir --no-build-isolation --no-deps -e .

CMD ["/bin/bash"]
