#!/bin/bash

uenv_image_find() {
    if [ -z $MYUENV ] ; then
        echo "prgenv-gnu/24.11:v2"
    else
        echo "$MYUENV" | tr , "\n"
    fi
}

# echo "$*"
$*
