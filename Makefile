export XAUTH_KEYS_="$(xauth list $HOST/unix:0)"
export XSOCK=/tmp/.X11-unix
export XAUTH=/tmp/.docker.xauth

build:
	touch ${XAUTH}
	docker build . -t p1 --build-arg "XAUTH_KEYS_=${XAUTH_KEYS_}"
	rm ${XAUTH} && touch ${XAUTH}
	xauth nlist ${DISPLAY} | sed -e 's/^..../ffff/' | xauth -f ${XAUTH} nmerge -

run:
	sudo docker run \
	-it \
	--rm \
	--net=host \
	-e DISPLAY=${DISPLAY} \
	-e XAUTHORITY=${XAUTH} \
	-v ${XSOCK}:${XSOCK}  \
	-v ${XAUTH}:${XAUTH} \
	p1

native-build:
	python3 -m pip install -r requirements.txt


native-run:
	python3 app/main.py
