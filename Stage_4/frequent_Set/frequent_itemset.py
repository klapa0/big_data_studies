from mrjob.job import MRJob

class MRFrequentItemSet(MRJob):

    def configure_args(self):
        super(MRFrequentItemSet, self).configure_args()
        # Zmieniono z add_passthrough_option na add_passthru_arg
        self.add_passthru_arg(
            '--min-support', 
            type=int, 
            default=2, 
            help='Minimum count to be frequent'
        )

    def mapper(self, _, line):
        items = line.strip().split(',')[1:]
        for item in items:
            yield item.strip(), 1

    def reducer(self, key, values):
        total_count = sum(values)
        # self.options automatycznie mapuje '--min-support' na 'min_support'
        min_support = self.options.min_support
        
        if total_count >= min_support:
            yield key, total_count

if __name__ == '__main__':
    MRFrequentItemSet.run()