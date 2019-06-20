#!/bin/bash

read_char() {
  # Cf. https://stackoverflow.com/questions/8725925/how-to-read-just-a-single-character-in-shell-script
  stty -icanon -echo
  eval "$1=\$(dd bs=1 count=1 2>/dev/null)"
  stty icanon echo
  echo "$char"
}

prompt() {
  echo -n "$1"
  read_char char
  while [ "$char" != 'y' ] && [ "$char" != 'Y' ] && [ "$char" != 'n' ] && [ "$char" != 'N' ]; do
    echo -n "$1"
    read_char char
  done
  if [ "$char" = 'y' ] || [ "$char" = 'Y' ]; then
    return 0
  else
    return 1
  fi
}
