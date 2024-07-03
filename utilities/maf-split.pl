#!/usr/bin/env perl
# Mario Stanke, 18 March 2024
# use like this:
# cat a.maf | ./maf-split.pl xx .maf 20 to split
# to obtain files xx000.maf, xx001.maf, ...
# such that each file has 20 or less records and cat xx*.maf = a.maf
use strict;
use warnings;
my ($prefix, $suffix, $records_per_file) = @ARGV;

local $/ = "\n\n"; 
my $fcount = 0;
my $rcount = 0;
my $output;
open ( $output, '>', sprintf("$prefix%03d$suffix", $fcount++ )) or die $!;

while ( my $chunk = <STDIN> ) {	
    print {$output} $chunk;
    $rcount++;
    if ($rcount == $records_per_file){
	close ( $output );
	open ( $output, '>', sprintf("$prefix%03d$suffix", $fcount++ )) or die $!;
	$rcount = 0;
    }
}
