#!/opt/homebrew/bin/python3

import sys
import json
import requests
import statistics
import bisect

BLOCK_LIMIT_MAINNET_20_WRITE_LENGTH = 15000000
BLOCK_LIMIT_MAINNET_20_WRITE_COUNT = 7750
BLOCK_LIMIT_MAINNET_20_READ_LENGTH = 100000000
BLOCK_LIMIT_MAINNET_20_READ_COUNT = 7750
BLOCK_LIMIT_MAINNET_20_RUNTIME = 5000000000

MAINNET_MAX_BLOCK_LEN = 2 * 1024 * 1024

PROPORTION_RESOLUTION = 10000
MICROSTACKS_PER_STACKS = 1000000

MAX_BLOCK_FILL = 6 * PROPORTION_RESOLUTION

RELAY_FEE_PER_BYTE = 1

WEIGHTED_QUANTILES = "WEIGHTED_QUANTILES"
UNWEIGHTED_QUANTILES = "UNWEIGHTED_QUANTILES"
FILL_WITH_RELAY_FEE = "FILL_WITH_RELAY_FEE"

ESTIMATE_EXP_WINDOW = "ESTIMATE_EXP_WINDOW"
EXP_DECAY = 0.5
ESTIMATE_MEDIAN_WINDOW = "ESTIMATE_MEDIAN_WINDOW"
WINDOW_SIZE = 5

METHODS = [ WEIGHTED_QUANTILES, FILL_WITH_RELAY_FEE, ] # ESTIMATE_EXP_WINDOW ]


with open('./transactions.json') as f:
    TRANSACTIONS = json.load(f)

def get_raw_tx(txid):
    if txid not in TRANSACTIONS:
        raise "Transaction not found -- store raw tx in transactions.json"
    return TRANSACTIONS[txid]

def scale_dimension(x, scale_by):
    return max(1.0, float(PROPORTION_RESOLUTION) * min(1.0, x / max(1.0, scale_by)))

def compute_metric(tx):
    tx_len = (float(len(get_raw_tx(tx['tx_id']))) - 2.0) / 2.0
    len_comp = scale_dimension(tx_len, MAINNET_MAX_BLOCK_LEN)
    if tx['tx_type'] == "token_transfer":
        return len_comp

    read_count_comp = scale_dimension(tx['execution_cost_read_count'], BLOCK_LIMIT_MAINNET_20_READ_COUNT)
    read_length_comp = scale_dimension(tx['execution_cost_read_length'], BLOCK_LIMIT_MAINNET_20_READ_LENGTH)
    write_count_comp = scale_dimension(tx['execution_cost_write_count'], BLOCK_LIMIT_MAINNET_20_WRITE_COUNT)
    write_length_comp = scale_dimension(tx['execution_cost_write_length'], BLOCK_LIMIT_MAINNET_20_WRITE_LENGTH)
    runtime_comp = scale_dimension(tx['execution_cost_runtime'], BLOCK_LIMIT_MAINNET_20_RUNTIME)

    return read_count_comp + read_length_comp + write_count_comp + write_length_comp + runtime_comp + len_comp

def weighted_quantiles(values, weights, req_percentiles):
    total_weight = float(sum(weights))  # S_N in weighted percentile formulae
    sorted_pairs = sorted(zip(values, weights))

    cumulative_weight = 0
    percentiles = [] # array will hold p_n
    for (value, weight) in sorted_pairs:
        cumulative_weight += weight # cumulative_weight is now S_n in weighted percentile formulae
        percentile_n = (cumulative_weight - weight/2.0)/total_weight
        percentiles.append(percentile_n)

    # percentiles array holds the [0, 1] values corresponding to the percentile of each
    #  input value. to find the estimated value that corresponds to the requested percentiles,
    #  we linearly interpolate between the values.
    values = [ v for (v, w) in sorted_pairs ]
    outputted_values = []
    for target_percentile in req_percentiles:
        if target_percentile < 0 or target_percentile > 1:
            raise "Percentile out of range"
        k_1 = bisect.bisect_right(percentiles, target_percentile)
        if k_1 == 0: # percentile is lower than minimum percentile in values
            outputted_values.append(values[0])
            continue
        if k_1 >= len(values):
            outputted_values.append(values[-1])
            continue
        k = k_1 - 1
        v = values[k] + (((target_percentile - percentiles[k]) / (percentiles[k_1] - percentiles[k])) * (values[k_1] - values[k]))
        outputted_values.append(v)
    return outputted_values

def simulate_estimator(block_stats):
    state = None

    log = []
    for (block, (low, middle, high)) in block_stats:
        if ESTIMATE_EXP_WINDOW in METHODS:
            if state is None:
                state = (low, middle, high)
            else:
                state = (state[0] * EXP_DECAY + (low * (1-EXP_DECAY)),
                         state[1] * EXP_DECAY + (middle * (1-EXP_DECAY)),
                         state[2] * EXP_DECAY + (high * (1-EXP_DECAY)))
            log.append((block, state))
        elif ESTIMATE_MEDIAN_WINDOW in METHODS:
            if state is None:
                state = ([low], [middle], [high])
            else:
                if len(state[0]) >= WINDOW_SIZE:
                    state[0].pop(0)
                    state[1].pop(0)
                    state[2].pop(0)
                state[0].append(low)
                state[1].append(middle)
                state[2].append(high)
            log.append((block, (statistics.median(state[0]),
                                statistics.median(state[1]),
                                statistics.median(state[2]))))
        else:
            raise "No estimator behavior specified"

    return log


def main(start_block, end_block):
    results = {}
    block_stats = []
    max_fee_rate = 0
    max_fee_rate_type = ""

    for block in range(start_block, end_block + 1):
        filename = "./blocks/block_%s.json" % block
        with open(filename) as f:
            data = json.load(f)
        transactions = data['results']
        if data['limit'] < data['total']:
            filename = "./blocks/block_%s.1.json" % block
            with open(filename) as f:
                data = json.load(f)
            transactions.extend(data['results'])
        block_fee_rates = []
        block_metrics = []
        for tx in transactions:
            metric = compute_metric(tx)
            fee = tx['fee_rate'] # poorly named API field
            fee_rate = float(fee) / metric
            tx_type = tx['tx_type']
            if tx_type == "coinbase":
                continue
            if tx_type in results:
                results[tx_type].append(fee_rate)
            else:
                results[tx_type] = [fee_rate]

            if fee_rate > max_fee_rate:
                max_fee_rate_type = "%s => %s @ %s" % (tx_type, fee, metric)

            max_fee_rate = max(max_fee_rate, fee_rate)

            block_fee_rates.append(fee_rate)
            block_metrics.append(metric)

        if len(block_fee_rates) >= 1:
            if FILL_WITH_RELAY_FEE in METHODS:
                # Only simulate block fill if the block didn't fill at least one dimension
                if sum(block_metrics) < PROPORTION_RESOLUTION:
                    leftover = PROPORTION_RESOLUTION - sum(block_metrics)
                    block_fee_rates.append(1)
                    block_metrics.append(leftover)

            if WEIGHTED_QUANTILES in METHODS:
                quants = weighted_quantiles(block_fee_rates, block_metrics, [0.05, 0.5, .95])
                block_stats.append((block, (quants[0], quants[1], quants[2])))
            elif UNWEIGHTED_QUANTILES in METHODS:
                highest_index = len(block_fee_rates) - max(1, int(len(block_fee_rates) / 20))
                median_index = int(len(block_fee_rates) / 2)
                lowest_index = int(len(block_fee_rates) / 20)

                block_fee_rates.sort()

                high = block_fee_rates[highest_index]
                low = block_fee_rates[lowest_index]
                median = block_fee_rates[median_index]
                block_stats.append((block, (low, median, high)))
            else:
                raise "No quantile method specified"

    block_stats.sort()
    print()
    print(" METHODS: %s" % METHODS)
    print()

    print()
    print(" ==== MAX FEE RATE TX ====")
    print()
    print(max_fee_rate, max_fee_rate_type)
    print()
    print(" ==== TRANSACTION TYPES ====")
    print()
    print("type,\t\tmean,\t\tmedian,\t\t5pp,\t\t95pp")
    print("--------------------------------------------------------------------")
    for (tx_type, rates) in results.items():
        high = statistics.quantiles(rates, n=20, method='inclusive')[-1]
        low = statistics.quantiles(rates, n=20, method='inclusive')[0]
        print("%s,\t%f,\t%f,\t%f,\t%f" % (tx_type, statistics.mean(rates), statistics.median(rates), low, high))
    print()
    print(" ==== IMPLIED STACK-STX COST ====")
    print()
    print("type,\t\tw/ low,\t\tw/ middle,\tw/ high")
    print("--------------------------------------------------------------------")
    for (tx_type, rates) in results.items():
        high = statistics.quantiles(rates, n=20, method='inclusive')[-1]
        low = statistics.quantiles(rates, n=20, method='inclusive')[0]
        median = statistics.median(rates)
        print("%s,\t%f,\t%f,\t%f" % (tx_type,
                                  low * 317.0 / MICROSTACKS_PER_STACKS,
                                  median * 317.0 / MICROSTACKS_PER_STACKS,
                                  high * 317.0 / MICROSTACKS_PER_STACKS))

    max_middle = (0, None)
    max_high = (0, None)
    max_low = (0, None)
    for (b, bstat) in block_stats:
        high = bstat[2]
        low = bstat[0]
        median = bstat[1]
        if high > max_high[0]:
            max_high = (high, (b, low, median, high))
        if median > max_middle[0]:
            max_middle = (median, (b, low, median, high))
        if low > max_low[0]:
            max_low = (low, (b, low, median, high))

    if ESTIMATE_EXP_WINDOW in METHODS or ESTIMATE_MEDIAN_WINDOW in METHODS:
        print()
        print(" ==== ESTIMATOR STATS ====")
        print()

        estimator_log = simulate_estimator(block_stats)
        for (b, bstat) in estimator_log:
            low = bstat[0]
            median = bstat[1]
            high = bstat[2]
            print("%s, %f, %f, %f" % (b, low, median, high))
    else:
        print()
        print(" ==== BLOCK STATS ====")
        print()
        for (b, bstat) in block_stats:
            low = bstat[0]
            median = bstat[1]
            high = bstat[2]
            print("%s, %f, %f, %f" % (b, low, median, high))

    # print()
    # print(" ==== MAX HIGH BLOCK ====")
    # print("%s, %f, %f, %f" % max_high[1])
    # print(" ==== MAX MIDDLE BLOCK ====")
    # print("%s, %f, %f, %f" % max_middle[1])
    # print(" ==== MAX LOW BLOCK ====")
    # print("%s, %f, %f, %f" % max_low[1])
    # print()
    # print(" ==== Implied stack-stx cost ====")
    # print()
    # stack_stx_multiple = 380.0 / MICROSTACKS_PER_STACKS
    # print("%s, %f, %f, %f" % (max_high[1][0],
    #                           max_high[1][1] * stack_stx_multiple,
    #                           max_high[1][2] * stack_stx_multiple,
    #                           max_high[1][3] * stack_stx_multiple))
    # print("%s, %f, %f, %f" % (max_middle[1][0],
    #                           max_middle[1][1] * stack_stx_multiple,
    #                           max_middle[1][2] * stack_stx_multiple,
    #                           max_middle[1][3] * stack_stx_multiple))
    # print("%s, %f, %f, %f" % (max_low[1][0],
    #                           max_low[1][1] * stack_stx_multiple,
    #                           max_low[1][2] * stack_stx_multiple,
    #                           max_low[1][3] * stack_stx_multiple))


    print()

main(int(sys.argv[1]), int(sys.argv[2]))
