clear
reset
set key off
set border 3

# Add a vertical dotted line at x=0 to show centre (mean) of distribution.
set yzeroaxis

# We want a small gap between solid (filled-in) bars.
set boxwidth 0.8 relative
set style fill solid 1.0

bin_number(x) = floor(x/bin_width)

rounded(x) = bin_width * ( bin_number(x) + 0.5 )

plot 'dataset.dat' using (rounded($1)):(1) smooth frequency with boxes