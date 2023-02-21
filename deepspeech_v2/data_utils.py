import torchaudio
import torch
import re


## comment out for base phoneme    

class TextTransform:
    """Maps phonemes to integers and vice versa"""
    def __init__(self):
        self.phonome_map_str = """
                                a 1
                                ã 2
                                b 3
                                bʰ 4
                                c 5
                                cʰ 6
                                d 7
                                dʰ 8
                                d̪ 9
                                d̪ʰ 10
                                e 11
                                ẽ 12
                                g 13
                                gʰ 14
                                h 15
                                i 16
                                ĩ 17
                                i̯ 18
                                k 19
                                kʰ 20
                                l 21
                                m 22
                                n 23
                                o 24
                                õ 25
                                o̯ 26
                                p 27
                                pʰ 28
                                r 29
                                s 30
                                t 31
                                tʰ 32
                                t̪ 33
                                t̪ʰ 34
                                u 35
                                ũ 36
                                u̯ 37
                                æ 38
                                æ̃ 39
                                ŋ 40
                                ɔ 41
                                ɔ̃ 42
                                ɟ 43
                                ɟʰ 44
                                ɽ 45
                                ɽʰ 46
                                ʃ 47
                                ʲ 48
                                ʷ 49
                                
                                """
        self.phone_map = {}
        self.index_map = {}
        for line in self.phonome_map_str.strip().split('\n'):
            ch, index = line.split()
            self.phone_map[ch] = int(index)
            self.index_map[int(index)] = ch
        # self.index_map[50] = '@'
        # self.phone_map['@'] = 50
        
    
    def remove_punctuations(self, text):
        # define punctuation
        regex =  r"[!\"#\$%।\'\(\)\–\*\+,-\./:‘’;<=>\?@\[\\\]\^_`{\|}~]"
        
        subst = ""

        result = re.sub(regex, subst, text, 0, re.MULTILINE)
        return result
    
    def remove_english_letters_and_numbers(self, text):
        return re.sub(r'[A-Za-z0-9]+[ \t]*', r'', text)
        
    def text_to_int(self, text):
        """ Use a phone map and convert phoneme sequence to an integer sequence """
        phone_list = text.split('_2')
        phone_list.pop()
        int_sequence = []
        # print(phone_list)
        for phone_per_word in phone_list:
            phone_per_word = phone_per_word.lstrip()
            phone_per_word = phone_per_word.replace("_1", "")
            phone = ""
            for i in range(len(phone_per_word)):
                if i == len(phone_per_word) - 1:
                    phone += phone_per_word[i]
                        # exit()
                    ch = self.phone_map[phone]
                    int_sequence.append(ch)
                elif phone_per_word[i] != " ":
                    phone += phone_per_word[i]
                else:
                    ch = self.phone_map[phone]
                    int_sequence.append(ch)
                    phone = ""
            int_sequence.append(self.phone_map['@'])
        # print(int_sequence)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('', ' ')

if __name__ == "__main__":
    demo = TextTransform()
    # # print(demo.phonome_map_dict)
    # int_seq = demo.text_to_int(re.sub(r'[A-Za-z0-9]+[ \t]*', r'', 'যুক্তরাষ্ট্র ও রাশিয়ার মধ্যে নতুন START8,2,234,, frOm Th চুক্তির ভবিষ্যৎ নিয়েও কিছু বলবেন আশা করি'))
    int_seq = demo.text_to_int('যুক্তরাষ্ট্র ও রাশিয়ার মধ্যে নতুন START8,2,234,, frOm Th চুক্তির ভবিষ্যৎ নিয়েও কিছু বলবেন আশা করি')

  
## comment out for positional phoneme
# class TextTransform:
#     """Maps phonemes to integers and vice versa"""
#     def __init__(self):
        
#         self.phonome_map_str = """
#                                 a 1
#                                 ã 2
#                                 b 3
#                                 bʰ 4
#                                 c 5
#                                 cʰ 6
#                                 d 7
#                                 dʰ 8
#                                 d̪ 9
#                                 d̪ʰ 10
#                                 e 11
#                                 ẽ 12
#                                 g 13
#                                 gʰ 14
#                                 h 15
#                                 i 16
#                                 ĩ 17
#                                 i̯ 18
#                                 k 19
#                                 kʰ 20
#                                 l 21
#                                 m 22
#                                 n 23
#                                 o 24
#                                 õ 25
#                                 o̯ 26
#                                 p 27
#                                 pʰ 28
#                                 r 29
#                                 s 30
#                                 t 31
#                                 tʰ 32
#                                 t̪ 33
#                                 t̪ʰ 34
#                                 u 35
#                                 ũ 36
#                                 u̯ 37
#                                 æ 38
#                                 æ̃ 39
#                                 ŋ 40
#                                 ɔ 41
#                                 ɔ̃ 42
#                                 ɟ 43
#                                 ɟʰ 44
#                                 ɽ 45
#                                 ɽʰ 46
#                                 ʃ 47
#                                 ʲ 48
#                                 ʷ 49
#                                 a_1 50
#                                 ã_1 51
#                                 b_1 52
#                                 bʰ_1 53
#                                 c_1 54
#                                 cʰ_1 55
#                                 d_1 56
#                                 dʰ_1 57
#                                 d̪_1 58
#                                 d̪ʰ_1 59
#                                 e_1 60
#                                 ẽ_1 61
#                                 g_1 62
#                                 gʰ_1 63
#                                 h_1 64
#                                 i_1 65
#                                 ĩ_1 66
#                                 i̯_1 67
#                                 k_1 68
#                                 kʰ_1 69
#                                 l_1 70
#                                 m_1 71
#                                 n_1 72
#                                 o_1 73
#                                 õ_1 74
#                                 o̯_1 75
#                                 p_1 76
#                                 pʰ_1 77
#                                 r_1 78
#                                 s_1 79
#                                 t_1 80
#                                 tʰ_1 81
#                                 t̪_1 82
#                                 t̪ʰ_1 83
#                                 u_1 84
#                                 ũ_1 85
#                                 u̯_1 86
#                                 æ_1 87
#                                 æ̃_1 88
#                                 ŋ_1 89
#                                 ɔ_1 90
#                                 ɔ̃_1 91
#                                 ɟ_1 92
#                                 ɟʰ_1 93
#                                 ɽ_1 94
#                                 ɽʰ_1 95
#                                 ʃ_1 96
#                                 ʲ_1 97
#                                 ʷ_1 98
#                                 a_2 99
#                                 ã_2 100
#                                 b_2 101
#                                 bʰ_2 102
#                                 c_2 103
#                                 cʰ_2 104
#                                 d_2 105
#                                 dʰ_2 106
#                                 d̪_2 107
#                                 d̪ʰ_2 108
#                                 e_2 109
#                                 ẽ_2 110
#                                 g_2 111
#                                 gʰ_2 112
#                                 h_2 113
#                                 i_2 114
#                                 ĩ_2 115
#                                 i̯_2 116
#                                 k_2 117
#                                 kʰ_2 118
#                                 l_2 119
#                                 m_2 120
#                                 n_2 121
#                                 o_2 122
#                                 õ_2 123
#                                 o̯_2 124
#                                 p_2 125
#                                 pʰ_2 126
#                                 r_2 127
#                                 s_2 128
#                                 t_2 129
#                                 tʰ_2 130
#                                 t̪_2 131
#                                 t̪ʰ_2 132
#                                 u_2 133
#                                 ũ_2 134
#                                 u̯_2 135
#                                 æ_2 136
#                                 æ̃_2 137
#                                 ŋ_2 138
#                                 ɔ_2 139
#                                 ɔ̃_2 140
#                                 ɟ_2 141
#                                 ɟʰ_2 142
#                                 ɽ_2 143
#                                 ɽʰ_2 144
#                                 ʃ_2 145
#                                 ʲ_2 146
#                                 ʷ_2 147
                                
#                                 """                        
                                
                                
#         self.phone_map = {}
#         self.index_map = {}
        
#         for line in self.phonome_map_str.strip().split('\n'):
#             ch, index = line.split()
#             self.phone_map[ch] = int(index)
#             self.index_map[int(index)] = ch
            
#         # self.index_map[50] = '@'
#         # self.phone_map['@'] = 50
        
    
#     def remove_punctuations(self, text):
#         # define punctuation
#         regex =  r"[!\"#\$%।\'\(\)\–\*\+,-\./:‘’;<=>\?@\[\\\]\^_`{\|}~]"
        
#         subst = ""

#         result = re.sub(regex, subst, text, 0, re.MULTILINE)
#         return result
    
#     def remove_english_letters_and_numbers(self, text):
#         return re.sub(r'[A-Za-z0-9]+[ \t]*', r'', text)
        
        
#     def text_to_int(self, text):
#         """ Use a phone map and convert phoneme sequence to an integer sequence """
#         phone_list = text.split('_2')  # #re.split(r"(_2)", phoneme_sequence)  
#         phone_list = "_2E".join(phone_list)
#         phone_list = phone_list.split('E')
#         phone_list.pop()
#         ##phone_list = text.split('_2')
#         ##phone_list.pop()
        
#         int_sequence = []
#         # print(phone_list)
        
#         for phone_per_word in phone_list:
#             phone_per_word = phone_per_word.lstrip()
#             # phone_per_word = phone_per_word.replace("_1", "")
            
#             phone = ""
            
#             for i in range(len(phone_per_word)):
#                 if i == len(phone_per_word) - 1:
#                     phone += phone_per_word[i]
#                     ch = self.phone_map[phone]
#                     int_sequence.append(ch)
                    
#                 elif phone_per_word[i] != " ":
#                     phone += phone_per_word[i]
                     
#                 else:
#                     ch = self.phone_map[phone]
#                     int_sequence.append(ch)
                    
#                     phone = ""
        
#         return int_sequence


#     def int_to_text(self, labels):
#         """ Use a character map and convert integer labels to an text sequence """
#         string = []
#         for i in labels:
#             string.append(self.index_map[i])
#         return ''.join(string).replace('', ' ')
    


# if __name__ == "__main__":
    
#     demo = TextTransform()
#     # # print(demo.phonome_map_dict)
#     # int_seq = demo.text_to_int(re.sub(r'[A-Za-z0-9]+[ \t]*', r'', 'যুক্তরাষ্ট্র ও রাশিয়ার মধ্যে নতুন START8,2,234,, frOm Th চুক্তির ভবিষ্যৎ নিয়েও কিছু বলবেন আশা করি'))
    
#     # int_seq = demo.text_to_int('যুক্তরাষ্ট্র ও রাশিয়ার মধ্যে নতুন START8,2,234,, frOm Th চুক্তির ভবিষ্যৎ নিয়েও কিছু বলবেন আশা করি')
#     int_seq = demo.text_to_int('s_1 i d̪ d̪ʰ a n t̪ o_2 n_1 e o̯ ʷ a r_2 p_1 e cʰ o n e_2 cʰ_1 i l e n_2 ʃ_1 o ɟ i b_2 o_1 ʷ a ɟ e d̪_2 ɟ_1 ɔ ʲ_2')
    
    
#     print('Integer sequeunce:', int_seq)
