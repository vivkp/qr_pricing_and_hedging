BEGIN {
    FS = ","
    OFS = ","
    low = anchor_spot - range_points
    high = anchor_spot + range_points
}
function trim(value) {
    sub(/^[ \t\r]+/, "", value)
    sub(/[ \t\r]+$/, "", value)
    return value
}
NR == 1 {
    for (i = 1; i <= NF; i++) column[trim($i)] = i
    required["strike_price"] = 1
    required["option_type"] = 1
    required["expiry"] = 1
    required["timestamp"] = 1
    for (name in required) {
        if (!(name in column)) {
            print "Missing required CSV column: " name > "/dev/stderr"
            exit 2
        }
    }
    print $0
    next
}
{
    if (index($0, "\"") > 0) {
        quoted_rows++
        next
    }
    if (trim($(column["option_type"])) != "CE") next
    if (substr(trim($(column["expiry"])), 1, 10) != expiry_date) next
    quote_date = substr(trim($(column["timestamp"])), 1, 10)
    if (quote_date < start_date || quote_date > end_date) next
    strike = trim($(column["strike_price"])) + 0
    if (strike < low || strike > high) next
    print $0
    kept++
}
END {
    printf "Kept %d data rows; skipped %d rows containing quote characters.\n", kept, quoted_rows > "/dev/stderr"
}
