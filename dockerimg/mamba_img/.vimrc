set nocompatible              " required
filetype off                  " required

" set the runtime path to include Vundle and initialize
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()

" alternatively, pass a path where Vundle should install plugins
"call vundle#begin('~/some/path/here')

" let Vundle manage Vundle, required
Plugin 'gmarik/Vundle.vim'
" Plugin 'tmhedberg/SimpylFold'
" let g:SimpylFold_docstring_preview=1
" Plugin 'vim-scripts/indentpython.vim'
Bundle 'Valloric/YouCompleteMe'
let g:ycm_autoclose_preview_window_after_completion=1
map <leader>g  :YcmCompleter GoToDefinitionElseDeclaration<CR>

" Plugin 'vim-syntastic/syntastic'
" Plugin 'nvie/vim-flake8'
" let python_highlight_all=1
" syntax on

"Plugin 'altercation/vim-colors-solarized'
"syntax enable
"set background=dark
"colorscheme solarized


" add all your plugins here (note older versions of Vundle
" used Bundle instead of Plugin)

" ...

" All of your Plugins must be added before the following line
call vundle#end()            " required
filetype plugin indent on    " required

"split navigations
nnoremap <C-J> <C-W><C-J>
nnoremap <C-K> <C-W><C-K>
nnoremap <C-L> <C-W><C-L>
nnoremap <C-H> <C-W><C-H>

" Enable folding
" set foldmethod=indent
" set foldlevel=99
" Enable folding with the spacebar
" nnoremap <space> za


" add the proper PEP8 indentation
au BufNewFile,BufRead *.py
    \ set tabstop=4 |
    \ set softtabstop=4 |
    \ set shiftwidth=4 |
"    \ set textwidth=79 |
    \ set expandtab |
    \ set autoindent |
    \ set fileformat=unix

" flagging unnecessary white spaces
" au BufRead,BufNewFile *.py,*.pyw,*.c,*.h match BadWhitespace /\s\+$/

" UTF8 support
set encoding=utf-8

" line number
set nu

" comment color
highlight Comment ctermfg=green


" remove trailing sapce
nnoremap <silent> <F5> :let _s=@/ <Bar> :%s/\s\+$//e <Bar> :let @/=_s <Bar> :nohl <Bar> :unlet _s <CR>
" show trailing space
autocmd Syntax * syn match ExtraWhitespace /\s\+$\| \+\ze\t/
set cursorline 


" disable auto-folding
" set wrap
set linebreak
set nolist  " list disables linebreak
set wm=0 " wrapmargin
set fo=cq "formatoption -=t
set tw=0 " textwidth


" Press Space to turn off highlighting and clear any message already
" displayed
set hlsearch
nnoremap <silent> <Space> :nohlsearch<Bar>:echo<CR>

