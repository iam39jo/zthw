/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

// do not attempt to compile this file with any other compiler
#ifdef __CUDACC__


#include <limits>
#include <komrade/detail/util/static.h>
#include <komrade/sorting/detail/device/cuda/stable_radix_sort_bits.h>

namespace komrade
{

namespace sorting
{

namespace detail
{

namespace device
{

namespace cuda
{

//////////////////
// 8 BIT TYPES //
//////////////////

template <typename KeyType>
void stable_radix_sort_key_small_dev(KeyType * keys, unsigned int num_elements)
{
    // encode the small types in 32-bit unsigned ints
    komrade::device_ptr<unsigned int> full_keys = komrade::device_malloc<unsigned int>(num_elements);

    komrade::transform(komrade::device_ptr<KeyType>(keys), 
                       komrade::device_ptr<KeyType>(keys + num_elements),
                       full_keys,
                       encode_uint<KeyType>());

    // sort the 32-bit unsigned ints
    stable_radix_sort_key_dev((unsigned int *) full_keys.get(), num_elements);
    
    // decode the 32-bit unsigned ints
    komrade::transform(full_keys,
                       full_keys + num_elements,
                       komrade::device_ptr<KeyType>(keys),
                       decode_uint<KeyType>());

    // release the temporary array
    komrade::device_free(full_keys);
}

template <typename KeyType>
void stable_radix_sort_key_dev(KeyType * keys, unsigned int num_elements,
                               komrade::detail::util::Int2Type<1>)
{
    stable_radix_sort_key_small_dev(keys, num_elements);
}


//////////////////
// 16 BIT TYPES //
//////////////////

    
template <typename KeyType>
void stable_radix_sort_key_dev(KeyType * keys, unsigned int num_elements,
                               komrade::detail::util::Int2Type<2>)
{
    stable_radix_sort_key_small_dev(keys, num_elements);
}


//////////////////
// 32 BIT TYPES //
//////////////////

template <typename KeyType> 
void stable_radix_sort_key_dev(KeyType * keys, unsigned int num_elements,
                               komrade::detail::util::Int2Type<4>,
                               komrade::detail::util::Bool2Type<true>,   
                               komrade::detail::util::Bool2Type<false>)  // uint32
{
    radix_sort((unsigned int *) keys, num_elements, encode_uint<KeyType>(), encode_uint<KeyType>());
}

template <typename KeyType> 
void stable_radix_sort_key_dev(KeyType * keys, unsigned int num_elements,
                               komrade::detail::util::Int2Type<4>,
                               komrade::detail::util::Bool2Type<true>,
                               komrade::detail::util::Bool2Type<true>)   // int32
{
    // find the smallest value in the array
    KeyType min_val = komrade::reduce(komrade::device_ptr<KeyType>(keys),
                                      komrade::device_ptr<KeyType>(keys + num_elements),
                                      (KeyType) 0,
                                      komrade::minimum<KeyType>());

    if(min_val < 0)
        //negatives present, sort all 32 bits
        radix_sort((unsigned int*) keys, num_elements, encode_uint<KeyType>(), decode_uint<KeyType>(), 32);
    else
        //all keys are positive, treat keys as unsigned ints
        radix_sort((unsigned int *) keys, num_elements, encode_uint<KeyType>(), encode_uint<KeyType>());
}

template <typename KeyType> 
void stable_radix_sort_key_dev(KeyType * keys, unsigned int num_elements,
                               komrade::detail::util::Int2Type<4>,
                               komrade::detail::util::Bool2Type<false>,
                               komrade::detail::util::Bool2Type<true>)  // float32
{
    radix_sort((unsigned int*) keys, num_elements, encode_uint<KeyType>(), decode_uint<KeyType>(), 32);
}

template <typename KeyType>
void stable_radix_sort_key_dev(KeyType * keys, unsigned int num_elements,
                               komrade::detail::util::Int2Type<4>)
{
    stable_radix_sort_key_dev(keys, num_elements,
                              komrade::detail::util::Int2Type<4>(),
                              komrade::detail::util::Bool2Type<std::numeric_limits<KeyType>::is_exact>(),
                              komrade::detail::util::Bool2Type<std::numeric_limits<KeyType>::is_signed>());
}

//////////////////
// 64 BIT TYPES //
//////////////////

template <typename KeyType,
          typename LowerBits, typename UpperBits, 
          typename LowerBitsExtractor, typename UpperBitsExtractor>
void stable_radix_sort_key_large_dev(KeyType * keys, unsigned int num_elements,
                                     LowerBitsExtractor extract_lower_bits,
                                     UpperBitsExtractor extract_upper_bits)
{
    // first sort on the lower 32-bits of the keys
    komrade::device_ptr<unsigned int> partial_keys = komrade::device_malloc<unsigned int>(num_elements);
    komrade::transform(komrade::device_ptr<KeyType>(keys), 
                       komrade::device_ptr<KeyType>(keys + num_elements),
                       partial_keys,
                       extract_lower_bits);

    komrade::device_ptr<unsigned int> permutation = komrade::device_malloc<unsigned int>(num_elements);
    komrade::range(permutation, permutation + num_elements);
    
    stable_radix_sort_key_value_dev((LowerBits *) partial_keys.get(), permutation.get(), num_elements);

    // permute full keys so lower bits are sorted
    komrade::device_ptr<KeyType> permuted_keys = komrade::device_malloc<KeyType>(num_elements);
    komrade::gather(permuted_keys, 
                    permuted_keys + num_elements, 
                    permutation,
                    komrade::device_ptr<KeyType>(keys));
    
    // now sort on the upper 32 bits of the keys
    komrade::transform(permuted_keys, 
                       permuted_keys + num_elements,
                       partial_keys,
                       extract_upper_bits);
    komrade::range(permutation, permutation + num_elements);
    
    stable_radix_sort_key_value_dev((UpperBits *) partial_keys.get(), permutation.get(), num_elements);

    // store sorted keys
    komrade::gather(komrade::device_ptr<KeyType>(keys), 
                    komrade::device_ptr<KeyType>(keys + num_elements),
                    permutation,
                    permuted_keys);

    komrade::device_free(partial_keys);
    komrade::device_free(permutation);
    komrade::device_free(permuted_keys);
}

    
template <typename KeyType>
void stable_radix_sort_key_dev(KeyType * keys, unsigned int num_elements,
                               komrade::detail::util::Int2Type<8>,
                               komrade::detail::util::Bool2Type<true>,
                               komrade::detail::util::Bool2Type<false>)  // uint64
{
    stable_radix_sort_key_large_dev<KeyType, unsigned int, unsigned int, lower_32_bits<KeyType>, upper_32_bits<KeyType> >
        (keys, num_elements, lower_32_bits<KeyType>(), upper_32_bits<KeyType>());
}

template <typename KeyType>
void stable_radix_sort_key_dev(KeyType * keys, unsigned int num_elements,
                               komrade::detail::util::Int2Type<8>,
                               komrade::detail::util::Bool2Type<true>,
                               komrade::detail::util::Bool2Type<true>)   // int64
{
    stable_radix_sort_key_large_dev<KeyType, unsigned int, int, lower_32_bits<KeyType>, upper_32_bits<KeyType> >
        (keys, num_elements, lower_32_bits<KeyType>(), upper_32_bits<KeyType>());
}

template <typename KeyType>
void stable_radix_sort_key_dev(KeyType * keys, unsigned int num_elements,
                               komrade::detail::util::Int2Type<8>,
                               komrade::detail::util::Bool2Type<false>,
                               komrade::detail::util::Bool2Type<true>)  // float64
{
    typedef unsigned long long uint64;
    stable_radix_sort_key_large_dev<uint64, unsigned int, unsigned int, lower_32_bits<KeyType>, upper_32_bits<KeyType> >
        (reinterpret_cast<uint64 *>(keys), num_elements, lower_32_bits<KeyType>(), upper_32_bits<KeyType>());
}

template <typename KeyType>
void stable_radix_sort_key_dev(KeyType * keys, unsigned int num_elements,
                               komrade::detail::util::Int2Type<8>)
{
    stable_radix_sort_key_dev(keys, num_elements,
                              komrade::detail::util::Int2Type<8>(),
                              komrade::detail::util::Bool2Type<std::numeric_limits<KeyType>::is_exact>(),
                              komrade::detail::util::Bool2Type<std::numeric_limits<KeyType>::is_signed>());
}

template <typename KeyType> 
void stable_radix_sort_key_dev(KeyType * keys, unsigned int num_elements)
{
    // TODO assert is_pod

    // dispatch on sizeof(KeyType)
    stable_radix_sort_key_dev(keys, num_elements, komrade::detail::util::Int2Type<sizeof(KeyType)>());
}


} // end namespace cuda

} // end namespace device

} // end namespace detail

} // end namespace sorting

} // end namespace komrade

#endif // __CUDACC__

