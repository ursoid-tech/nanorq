use strict;
use warnings;
use Test::More;
use t::Util;

my ($stderr, $stdout) = run_prog('./t/00util/test_ops');
is($?, 0, 'schedule replay regressions exit cleanly');
like($stdout, qr/schedule replay regressions passed/,
     'workspace lifetime, permutations, and SIMD tails pass');
is($stderr, '', 'schedule replay regressions have no diagnostics');
done_testing();
