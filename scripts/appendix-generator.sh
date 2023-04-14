#!/bin/bash

filename="optimiser-appendix.py"

# An array of strings to loop through
strings=("Rb87Cs133" "K40Rb87" "Na23K40" "Na23Rb87" "Na23Cs133")

# Get the length of the array
len=${#strings[@]}

python3 "$filename"

# Loop through the array
for ((i=0; i<$len-1; i++)); do
  # Get the current and next strings
  current_string=${strings[$i]}
  next_string=${strings[$((i+1))]}

  # Use sed to find and replace the current string with the next string
  sed "s/$current_string/$next_string/g" "$filename" > output.file

  # Move the output file to the original filename for the next iteration
  mv output.file "$filename"

  # Run the updated Python script
  python3 "$filename"
done
