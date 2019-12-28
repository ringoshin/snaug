#
# The resulting file will become the input data source for ML text generation, hence 
# it should contain only story elements and little else.
# 
# This module will remove non-story elements from the raw game guide text data:
#     Lines that usually do not end with any punctuations like: .,?! and "'
#
#

in_filename = 'data/textgen_pathfinder_raw.txt'
out_filename = 'data/textgen_pathfinder.txt'

# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# return True if line is a valid story element
# by checking whether there is punctuation at the end
def valid_line(line):
    line = line.strip()
    if len(line) == 0:
        return False   # empty line
    elif line.split()[-1].lower() in ('combat!', 'ft.'):
        return False   # ends with "COMBAT! or Ft."
    else:
        return line[-1] in ",.?!\'\""

# load
doc = load_doc(in_filename)
lines = doc.split('\n')

print("\n>>> Cleaning '{}' now...".format(in_filename))
new_lines = []
for line in lines:
    if valid_line(line):
        new_lines.append(line)

# save new lines to file
save_doc(new_lines, out_filename)
print(">>> Done and saved to '{}'!\n".format(out_filename))