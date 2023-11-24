#!/bin/bash

sleep 5s
instruction_list=(
  "fly to the stern of the ship"
  "fly to the left side of flower"
  "turn left and go to the right side of tree"
  "move foward to reach the back of anvil"
  "keep roughly straight to the right of apple"
  "walk about a quarter of a circle around the apple"
  "pass the right and reach the back side of hydrant"
  "pass the left and reach the back side of stone"
)
echo instruction_list
for instruction in "${instruction_list[@]}"; do
  xte "str [prompt] $instruction"
  xte "keydown Shift_L" "key Return" "keyup Shift_L"
  xte "str please forget all previous commands."
  xte "key Return"
  sleep 5s
done
