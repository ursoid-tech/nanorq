use strict;
use warnings;
use Test::More;
use t::Util;

my ($timing, $output) = run_prog('./t/00util/test_oblas_swap');
is($?, 0, 'row swap regressions exit cleanly');
like($output, qr/row swap regressions passed/, 'disjoint self and unaligned swaps pass');
($timing, $output) = run_prog('./t/00util/test_oblas_kernels');
is($?, 0, 'SIMD kernel regressions exit cleanly');
like($output, qr/SIMD kernel regressions passed/, 'supported SIMD backends match independent GF256 reference');
done_testing();
