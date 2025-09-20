use std::{collections::VecDeque, ops::Range};

pub struct Handles<T> {
    pub items: Vec<T>,
    free_list: Vec<usize>,
}

impl<T> Handles<T> {
    #[inline]
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            free_list: Vec::new(),
        }
    }

    #[inline]
    pub fn track(&mut self, item: T) -> usize {
        if let Some(idx) = self.free_list.pop() {
            self.items[idx] = item;
            idx
        } else {
            let idx = self.items.len();
            self.items.push(item);
            idx
        }
    }

    #[inline]
    pub fn untrack(&mut self, idx: usize) {
        self.free_list.push(idx);
    }
}

pub struct HandlePool<T> {
    pub items: Vec<T>,
    free_list: Vec<usize>,
}

impl<T: Default> HandlePool<T> {
    #[inline]
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            free_list: Vec::new(),
        }
    }

    #[inline]
    fn find_free(&mut self) -> usize
    where
        T: Default,
    {
        if let Some(idx) = self.free_list.pop() {
            idx
        } else {
            let idx = self.items.len();
            self.items.push(T::default());
            idx
        }
    }

    #[inline]
    pub fn track<I: Fn(&mut T)>(&mut self, init: I) -> usize
    where
        T: Default,
    {
        let idx = self.find_free();
        init(&mut self.items[idx]);
        idx
    }

    #[inline]
    pub fn untrack(&mut self, idx: usize) {
        self.free_list.push(idx);
    }
}

#[derive(Default)]
pub struct MetaAlloc {
    pub range: Range<usize>,
    // TODO: generations for double free detection?
}

pub struct MetaAllocator {
    min_order: usize, // smallest allocation size: 2^(min_order)
    max_order: usize, // largest allocation size: 2^(max_order)
    free_lists: Vec<VecDeque<usize>>,
}

impl MetaAllocator {
    #[inline]
    pub fn new(size: usize, min_size: usize) -> Self {
        let size = size.next_power_of_two();
        let min_size = min_size.next_power_of_two();
        let max_order = size.trailing_zeros() as usize;
        let min_order = min_size.trailing_zeros() as usize;
        let mut free_lists = vec![VecDeque::new(); max_order + 1];
        free_lists[max_order].push_back(0);
        Self {
            min_order,
            max_order,
            free_lists,
        }
    }

    pub fn alloc(&mut self, size: usize) -> Option<MetaAlloc> {
        let order = size
            .max(self.min_order)
            .next_power_of_two()
            .trailing_zeros() as usize;
        if order > self.max_order {
            return None;
        }
        let mut found_offset = None;
        for cur_order in order..=self.max_order {
            if let Some(offset) = self.free_lists[cur_order].pop_front() {
                for split_order in (order..cur_order).rev() {
                    let buddy = offset + (1 << split_order);
                    self.free_lists[split_order].push_back(buddy);
                }
                found_offset = Some(offset);
            }
        }
        if let Some(offset) = found_offset {
            Some(MetaAlloc {
                range: offset..(offset + (1 << order)),
            })
        } else {
            None
        }
    }

    pub fn free(&mut self, alloc: MetaAlloc) {
        let order = alloc
            .range
            .len()
            .max(self.min_order)
            .next_power_of_two()
            .trailing_zeros() as usize;
        let mut cur_offset = alloc.range.start;
        let mut cur_order = order;
        while cur_order < self.max_order {
            let buddy = cur_offset ^ (1 << cur_order);
            if let Some(pos) = self.free_lists[cur_order]
                .iter()
                .position(|&off| off == buddy)
            {
                self.free_lists[cur_order].remove(pos);
                cur_offset = cur_offset.min(buddy);
                cur_order += 1;
            } else {
                break;
            }
        }
        self.free_lists[cur_order].push_back(cur_offset);
    }
}

pub struct BitMap {
    words: Vec<u64>,
    free_list: Vec<usize>,
}

impl BitMap {
    const BITS: usize = u64::BITS as usize;

    #[inline]
    pub fn new(size: usize) -> Self {
        let num_words = (size + (Self::BITS - 1)) / Self::BITS;
        Self {
            words: vec![0; num_words],
            free_list: Vec::new(),
        }
    }

    pub fn set_any(&mut self) -> Option<usize> {
        if let Some(idx) = self.free_list.pop() {
            let word_idx = idx / Self::BITS;
            let bit_idx = idx % Self::BITS;
            self.words[word_idx] |= 1 << bit_idx;
            Some(idx)
        } else {
            for (word_idx, word) in self.words.iter_mut().enumerate() {
                if *word == u64::MAX {
                    continue;
                }
                let bit_idx = word.trailing_ones() as usize;
                *word |= 1 << bit_idx;
                return Some((word_idx * Self::BITS) + bit_idx);
            }
            None
        }
    }

    #[inline]
    pub fn unset(&mut self, idx: usize) {
        let word_idx = idx / Self::BITS;
        let bit_idx = idx % Self::BITS;
        self.words[word_idx] &= !(1 << bit_idx);
        self.free_list.push(idx);
    }
}
