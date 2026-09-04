import pandas as pd


def convert_timestamp(df):
	df['timestamp'] = pd.to_datetime(
		df['timestamp'],
		format='ISO8601',
		utc=True
	)
	return df


def check_class_balance(df):
	class_distribution = df['label'].value_counts().sort_index()
	class_percentages = (class_distribution / len(df) * 100).round(2)
	balance_check = pd.DataFrame({
		'effectif': class_distribution,
		'pourcentage': class_percentages
	})

	print('\nRépartition des classes :')
	print(balance_check)

	imbalance_ratio = class_distribution.max() / class_distribution.min()
	print(f'\nRatio de déséquilibre : {imbalance_ratio:.2f}')


def calculate_malicious_percentage(df):
	malicious_count = (df['label'] == 1).sum()
	malicious_percentage = round(malicious_count / len(df) * 100, 2)
	print(f'\nPourcentage de flux malveillants : {malicious_percentage:.2f}%')


def count_source_ips(df):
	different_source_ips = df['src_ip'].nunique()
	print(f'\nNombre d\'IP sources différentes : {different_source_ips}')


def count_destinations(df):
	different_destinations = df['dst_ip'].nunique()
	print(f'\nNombre de destinations différentes : {different_destinations}')


def coefficient_of_variation(values):
	values = values.dropna()
	mean = values.mean()
	if values.empty or mean == 0:
		return 0
	return values.std(ddof=0) / mean


def recurrent_delta_ratio(values):
	values = values.dropna()
	if values.empty:
		return 0
	median = values.median()
	if median == 0:
		return 1
	return (values.sub(median).abs() <= median * 0.1).mean()


def aggregate_flows(df):
	df = df.copy()
	df['day'] = df['timestamp'].dt.date
	df = df.sort_values(['src_ip', 'dst_ip', 'timestamp'])
	df['delta_s'] = (
		df.groupby(['src_ip', 'dst_ip'])['timestamp']
		.diff()
		.dt.total_seconds()
	)
	df['is_night'] = (
		(df['timestamp'].dt.hour < 6) |
		(df['timestamp'].dt.hour >= 22)
	)

	aggregated_df = df.groupby(
		['day', 'src_ip', 'dst_ip'],
		as_index=False
	).agg(
		connection_count=('timestamp', 'size'),
		delta_mean_s=('delta_s', 'mean'),
		delta_std_s=('delta_s', lambda values: values.std(ddof=0)),
		delta_cv=('delta_s', coefficient_of_variation),
		recurrent_ratio=('delta_s', recurrent_delta_ratio),
		bytes_out_mean=('bytes_out', 'mean'),
		bytes_out_cv=('bytes_out', coefficient_of_variation),
		bytes_in_mean=('bytes_in', 'mean'),
		duration_mean_ms=('duration_ms', 'mean'),
		packets_out_mean=('packets_out', 'mean'),
		tls_ratio=('tls', 'mean'),
		night_ratio=('is_night', 'mean'),
		unique_ports=('dst_port', 'nunique'),
		label=('label', 'max')
	)
	aggregated_df[['delta_mean_s', 'delta_std_s']] = (
		aggregated_df[['delta_mean_s', 'delta_std_s']].fillna(0)
	)

	return aggregated_df


def main():
	df = pd.read_csv('network_flows.csv')
	df = convert_timestamp(df)
	check_class_balance(df)
	calculate_malicious_percentage(df)
	count_source_ips(df)
	count_destinations(df)

	aggregated_df = aggregate_flows(df)
	aggregated_df.to_csv('beacon_dataset.csv', index=False)
	print('\nDataset agrégé :')
	print(aggregated_df.head())
	print(f'\nNombre d\'observations ML : {len(aggregated_df)}')


if __name__ == '__main__':
	main()
