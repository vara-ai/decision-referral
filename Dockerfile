FROM python:3.8.10

WORKDIR /vara_dr
COPY requirements.txt /vara_dr/requirements.txt
RUN pip install -r requirements.txt

ENV PYTHONPATH="/vara_dr:$PYTHONPATH"

CMD ["/bin/bash"]
